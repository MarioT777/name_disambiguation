import logging
from os.path import join
from gensim.models import Word2Vec
import numpy as np
import random
from utils.cache import LMDBClient
from utils import data_utils
from utils.data_utils import Singleton
from utils import settings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
EMB_DIM = 100


@Singleton
class EmbeddingModel:

    def __init__(self, name="aminer"):
        self.model = None
        self.name = name

    def train(self, wf_name, size=EMB_DIM):
        data = []
        LMDB_NAME = 'pub_authors_pair.feature'  # (pid-j, author_feature)
        lc = LMDBClient(LMDB_NAME)
        author_cnt = 0
        with lc.db.begin() as txn:
            for k in txn.cursor():
                author_feature = data_utils.deserialize_embedding(k[1])  # 从K[1]中反序列化得到作者特征对象
                if author_cnt % 10000 == 0:
                    print(author_cnt, author_feature[0])
                author_cnt += 1
                random.shuffle(author_feature)  # 打乱作者特征
                # print(author_feature)
                data.append(author_feature)
        # 通过word2vec进行训练
        self.model = Word2Vec(
            data, size, window=5, min_count=5, workers=20,
        )  # 输入字符集，词向量维度100，窗口大小（当前词与目标词的最大距离），词频过滤值，训练并行数
        # 保存训练好的模型
        self.model.save(join(settings.EMB_DATA_DIR, '{}.emb'.format(wf_name)))  # 训练的结果保存到aminer.emb：{feature: emb}

    def load(self, name):
        self.model = Word2Vec.load(join(settings.EMB_DATA_DIR, '{}.emb'.format(name)))
        return self.model

    def project_embedding(self, tokens, idf=None):  # 输入特征集features，idf字典 {feature: idf}
        """
        weighted average of token embeddings
        对嵌入向量进行加权平均
        :param tokens: input words
        :param idf: IDF dictionary
        :return: obtained weighted-average embedding
        """
        if self.model is None:
            self.load(self.name)
            print('{} embedding model loaded'.format(self.name))
        vectors = []  # 向量集
        sum_weight = 0  # 权值和
        for token in tokens:  # 枚举一个特征单词
            if not token in self.model.wv:  # 如果token不在Word2Vec模型中，跳过
                continue
            weight = 1
            if idf and token in idf:  # token能在idf字典中查到
                weight = idf[token]  # 取出对应idf值
            v = self.model.wv[token] * weight  # 取出Word2Vec中对应的向量，并乘以权重
            vectors.append(v)  # 加入到向量集中
            sum_weight += weight  # 计算总权重
        if len(vectors) == 0:
            print('all tokens not in w2v models')
            # return np.zeros(self.model.vector_size)
            return None
        emb = np.sum(vectors, axis=0)  # 向量集所有向量之和
        emb /= sum_weight  # 除以总的权重和
        return emb  # 返回一个嵌入的低维向量，在word2vec得到100维向量映射后，利用idf进行加权平均的到特征嵌入Xi


if __name__ == '__main__':
    wf_name = 'aminer_pair'
    emb_model = EmbeddingModel.Instance()
    emb_model.train(wf_name)
    print('loaded')
