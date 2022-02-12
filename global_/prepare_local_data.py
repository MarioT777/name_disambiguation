from os.path import join
import os
import numpy as np
from numpy.random import shuffle
from global_.global_model import GlobalTripletModel
from utils.eval_utils import get_hidden_output
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

IDF_THRESHOLD = 32  # small data
# IDF_THRESHOLD = 10


def dump_inter_emb():
    """
    从训练的全局模型中取出隐藏层，给局部模型使用
    dump hidden embedding via trained global model for local model to use
    """
    LMDB_NAME = "author_100_pair.emb.weighted"  # {pid_j,Xi}
    lc_input = LMDBClient(LMDB_NAME)
    INTER_LMDB_NAME = 'author_triplets.emb'  # (pid_j, y)
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    global_model = GlobalTripletModel(data_scale=1000000)
    trained_global_model = global_model.load_triplets_model()  # 加载一个训练好的全局模型
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    # {name->aid->pid_j}
    for name in name_to_pubs_test:
        print('name', name)
        name_data = name_to_pubs_test[name]  # {aid:pid_j}
        embs_input = []  # 每篇论文的特征嵌入集
        pids = []  # 论文id集
        for i, aid in enumerate(name_data.keys()):
            if len(name_data[aid]) < 5:  # n_pubs of current author is too small
                continue
            for pid in name_data[aid]:
                cur_emb = lc_input.get(pid)  # 取出论文对应的嵌入Xi
                if cur_emb is None:
                    continue
                embs_input.append(cur_emb)
                pids.append(pid)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        for i, pid_ in enumerate(pids):
            lc_inter.set(pid_, inter_embs[i])


def gen_local_data(idf_threshold=10):
    """
    生成局部链接图，idf_threshold表示阈值，即相似度高于多少才构建边
    generate local data (including paper features and paper network) for each associated name
    :param idf_threshold: threshold for determining whether there exists an edge between two papers (for this demo we set 29)
    """
    name_to_pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf_pair.pkl')  # {feature:idf}
    INTER_LMDB_NAME = 'author_triplets.emb'  # 加载作者在triplet训练后的内部嵌入（pid_j,y） ？
    lc_inter = LMDBClient(INTER_LMDB_NAME)
    LMDB_AUTHOR_FEATURE = "pub_authors_pair.feature"  # (pid-j, author_feature)
    lc_feature = LMDBClient(LMDB_AUTHOR_FEATURE)
    graph_dir = join(settings.DATA_DIR, 'local', 'graph-{}'.format(idf_threshold))
    os.makedirs(graph_dir, exist_ok=True)
    for i, name in enumerate(name_to_pubs_test):
        print(i, name)
        cur_person_dict = name_to_pubs_test[name]  # {aid:pid_j}
        pids_set = set()  # 论文id集合，通过集合来去重
        pids = []  # 论文id列表
        pids2label = {}  # 论文id:作者id映射{pid:aid}

        # 生成每个待消歧姓名的局部数据（论文特征嵌入，论文对应的聚类实体），(pid_j, y, aid)
        wf_content = open(join(graph_dir, '{}_pubs_content.txt'.format(name)), 'w')
        for j, aid in cur_person_dict:
            items = cur_person_dict[aid]
            if len(items) < 5:
                continue
            for pid in items:
                pids2label[pid] = aid  # 论文标记{pid:aid}
                pids.append(pid)
        shuffle(pids)  # 打乱
        for pid in pids:
            cur_pub_emb = lc_inter.get(pid)  # 获得论文嵌入y
            if cur_pub_emb is not None:
                cur_pub_emb = list(map(str, cur_pub_emb))  # 将cur_pub_emb中的每个元素转换为str，并以列表的形式返回
                pids_set.add(pid)  # 去重
                wf_content.write('{}\t'.format(pid))  # 论文id
                wf_content.write('\t'.join(cur_pub_emb))  # 嵌入y
                wf_content.write('\t{}\n'.format(pids2label[pid]))  # pid: aid,论文对应的聚类实体
        wf_content.close()

        # generate network，构建局部链接图
        pids_filter = list(pids_set)
        n_pubs = len(pids_filter)  # 论文总数
        print('n_pubs', n_pubs)
        wf_network = open(join(graph_dir, '{}_pubs_network.txt'.format(name)), 'w')
        edges_num = 0
        for n in range(n_pubs-1):
            if n % 10 == 0:
                print(n)
            author_feature1 = set(lc_feature.get(pids_filter[n]))  # 取出论文n的作者特征
            print(author_feature1)
            for m in range(n+1, n_pubs):
                author_feature2 = set(lc_feature.get(pids_filter[m]))  # 取出论文m的作者特征
                common_features = author_feature1.intersection(author_feature2)  # 提取公共特征
                idf_sum = 0
                for f in common_features:
                    idf_sum += idf.get(f, idf_threshold)  # 计算idf和
                    # print(f, idf.get(f, idf_threshold))
                if idf_sum >= idf_threshold:  # 大于设定的阈值idf_threshold
                    wf_network.write('{}\t{}\n'.format(pids_filter[n], pids_filter[m]))  # 构建边，写入图网络文件（pid_j,pid_j）
                    edges_num += 1
        print('n_edges', edges_num)
        wf_network.close()


if __name__ == '__main__':
    dump_inter_emb()  # 提取内层嵌入
    gen_local_data(idf_threshold=IDF_THRESHOLD)  # 生成局部数据（同一作者的相关论文信息）和局部链接图
    print('done')
