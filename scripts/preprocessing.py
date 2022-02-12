from os.path import join
import codecs
import math
from collections import defaultdict as dd
from global_.embedding import EmbeddingModel
from datetime import datetime
from utils import data_utils
from utils import feature_utils
from utils import settings
from utils.cache import LMDBClient

start_time = datetime.now()


def dump_author_features_to_file():
    """
    提取作者特征保存到文件中
    generate author features by raw publication data and dump to files
    author features are defined by his/her paper attributes excluding the author's name
    """
    pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'author_yangbing.json')  # 原始数据pubs_raw.json
    print('n_papers', len(pubs_dict))  # 论文数量
    wf = codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_features_pair.txt'), 'w', encoding='utf-8')
    for i, pid in enumerate(pubs_dict):
        if i % 1000 == 0:
            print(i, datetime.now() - start_time)
        paper = pubs_dict[pid]
        if "title" not in paper or "authors" not in paper:
            continue
        if len(paper["authors"]) > 30:  # 合作者人数
            print(i, pid, len(paper["authors"]))
        if len(paper["authors"]) > 100:
            continue
        n_authors = len(paper.get('authors', []))  # 该论文作者数，dict.get(key, default=None)在字典中查询键值key，若不存在返回默认值default
        for j in range(n_authors):  # 枚举每一位作者
            if 'id' not in paper['authors'][j]:
                continue
            author_feature = feature_utils.extract_author_features(paper, j)  # 提取作者特征
            aid = '{}-{}'.format(pid, j)  # aid: pid-j
            wf.write(aid + '\t' + ' '.join(author_feature) + '\n')
    wf.close()


def dump_author_features_to_cache():
    """
    将作者特征导入cache中（lmdb本地数据库）
    dump author features to lmdb
    """
    LMDB_NAME = 'pub_authors_pair.feature'
    lc = LMDBClient(LMDB_NAME)
    with codecs.open(join(settings.GLOBAL_DATA_DIR, 'author_features_pair.txt'), 'r', encoding='utf-8') as rf:
        for i, line in enumerate(rf):
            if i % 1000 == 0:
                print('line', i)
            items = line.rstrip().split('\t')  # 删除末尾空格后，按'\t'分割  pid-j, author_feature
            pid_order = items[0]  # 提取论文序号pod_j
            author_features = items[1].split()  # 提取作者特征，分割为列表
            lc.set(pid_order, author_features)  # 导入本地数据库lmdb


def cal_feature_idf():
    """
    计算逆文本频率
    calculate word IDF (Inverse document frequency) using publication data
    """
    feature_dir = join(settings.DATA_DIR, 'global')
    counter = dd(int)  # 一种字典，比{}多一个，如果没有查询到的key，会返回int(0)
    cnt = 0
    LMDB_NAME = 'pub_authors_pair.feature'
    lc = LMDBClient(LMDB_NAME)  # 连接本地数据库
    author_cnt = 0
    with lc.db.begin() as txn:
        for k in txn.cursor():  # 遍历数据库中所有的记录 *
            features = data_utils.deserialize_embedding(k[1])  # 反序列化得到作者特征author_feature
            if author_cnt % 10000 == 0:
                print(author_cnt, features[0], counter.get(features[0]))
                # features[0] 类似"__NAME__yanjun_zhang" 是合作者的name_feature
            author_cnt += 1   # 作者总数
            for f in features:
                cnt += 1  # 特征的总数
                counter[f] += 1  # 每项特征出现的次数
    idf = {}
    for k in counter:
        idf[k] = math.log(cnt / counter[k])  # 计算每项特征对应的idf
    data_utils.dump_data(dict(idf), feature_dir, "feature_idf_pair.pkl")  # 写入 feature_idf.pkl中{feature: idf}


def dump_author_embs():
    """
    dump author embedding to lmdb
    author embedding is calculated by weighted-average of word vectors with IDF
    基于逆文本频率（IDF）对词向量进行加权平均得到特征嵌入
    """
    emb_model = EmbeddingModel.Instance()
    idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf_pair.pkl')  # {feature: idf}
    print('idf loaded')
    LMDB_NAME_FEATURE = 'pub_authors_pair.feature'  # (pid-j, author_feature)
    lc_feature = LMDBClient(LMDB_NAME_FEATURE)  # 连接作者特征lmdb
    LMDB_NAME_EMB = "author_100_pair.emb.weighted"  # (pid-j, x^-)
    lc_emb = LMDBClient(LMDB_NAME_EMB)
    cnt = 0
    with lc_feature.db.begin() as txn:
        for k in txn.cursor():  # 遍历作者特征
            if cnt % 1000 == 0:
                print('cnt', cnt, datetime.now() - start_time)
            cnt += 1
            pid_order = k[0].decode('utf-8')
            features = data_utils.deserialize_embedding(k[1])
            cur_emb = emb_model.project_embedding(features, idf)
            if cur_emb is not None:
                lc_emb.set(pid_order, cur_emb)


if __name__ == '__main__':
    """
    some pre-processing
    """
    dump_author_features_to_file()
    dump_author_features_to_cache()
    emb_model = EmbeddingModel.Instance()
    emb_model.train('aminer_pair')  # training word embedding model
    cal_feature_idf()
    dump_author_embs()
    print('done', datetime.now() - start_time)
