# -*- coding: utf-8 -*-

from os.path import join
import os
import multiprocessing as mp
import random
from datetime import datetime
from utils.cache import LMDBClient
from utils import data_utils
from utils import settings

LMDB_NAME = "author_100.emb.weighted"  # 链接数据库，取出每篇论文的特征嵌入(pid_j, xi)
lc = LMDBClient(LMDB_NAME)
start_time = datetime.now()

# 为训练全局模型， 生成三元组的训练集
"""
This class generates triplets of author embeddings to train global model
"""


class TripletsGenerator:
    name2pubs_train = {}  # 训练集原始数据
    name2pubs_test = {}  # 测试集原始数据
    names_train = None  # 训练集中的所有作者姓名集
    names_test = None  # 测试集中的所有作者姓名集
    n_pubs_train = None  # 训练集论文数量
    n_pubs_test = None  # 测试集论文数量
    pids_train = []  # 训练集下的论文集
    pids_test = []  # 测试集下的论文集
    n_triplets = 0  # 三元组数量
    batch_size = 100000

    def __init__(self, train_scale=10000):  # 构造函数
        self.prepare_data()
        self.save_size = train_scale
        self.idf = data_utils.load_data(settings.GLOBAL_DATA_DIR, 'feature_idf.pkl')  # {feature:idf}

    def prepare_data(self):
        self.name2pubs_train = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_train_500.json')  # name->aid->pid_j
        self.name2pubs_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'name_to_pubs_test_100.json')
        self.names_train = self.name2pubs_train.keys()  # 所有同名作者
        print('names train', len(self.names_train))
        self.names_test = self.name2pubs_test.keys()
        print('names test', len(self.names_test))
        assert not set(self.names_train).intersection(set(self.names_test))  # 返回两个集合都包含的元素
        for name in self.names_train:  # 枚举训练集中的作者
            name_pubs_dict = self.name2pubs_train[name]
            for aid in name_pubs_dict:
                self.pids_train += name_pubs_dict[aid]  # 待消歧论文集合
        random.shuffle(self.pids_train)  # 随机打乱
        self.n_pubs_train = len(self.pids_train)
        print('pubs2train', self.n_pubs_train)

        for name in self.names_test:
            name_pubs_dict = self.name2pubs_test[name]
            for aid in name_pubs_dict:
                self.pids_test += name_pubs_dict[aid]
        random.shuffle(self.pids_test)
        self.n_pubs_test = len(self.pids_test)
        print('pubs2test', self.n_pubs_test)

    def gen_neg_pid(self, not_in_pids, role='train'):
        """
        生成三元组损失的负样本
        :param not_in_pids: 目标论文集
        :return:
        """
        if role == 'train':
            sample_from_pids = self.pids_train
        else:
            sample_from_pids = self.pids_test
        while True:
            idx = random.randint(0, len(sample_from_pids) - 1)  # 从论文集中随机选取一篇不在目标论文集中的论文
            pid = sample_from_pids[idx]
            if pid not in not_in_pids:
                return pid

    def sample_triplet_ids(self, task_q, role='train', N_PROC=8):
        """
        生成三元组（generate triples from the paper set）
        :param role:
        :param task_q:
        :param N_PROC:
        :return:
        """
        n_sample_triplets = 0
        if role == 'train':
            names = self.names_train
            name2pubs = self.name2pubs_train  # name->aid->pid-j
        else:  # test
            names = self.names_test
            name2pubs = self.name2pubs_test
            self.save_size = 200000  # test save size
        # 从聚类实体aid对应的论文集中随机选取一篇论文作为anchor值
        for name in names:
            name_pubs_dict = name2pubs[name]
            for aid in name_pubs_dict:
                pub_items = name_pubs_dict[aid]  # 作者相关的所有论文pid_j
                if len(pub_items) == 1:
                    continue
                pids = pub_items
                cur_n_pubs = len(pids)
                random.shuffle(pids)  # 随机打乱
                for i in range(cur_n_pubs):
                    pid1 = pids[i]  # pid

                    # batch samples
                    n_samples_anchor = min(6, cur_n_pubs)
                    idx_pos = random.sample(range(cur_n_pubs), n_samples_anchor)
                    for ii, i_pos in enumerate(idx_pos):
                        if i_pos != i:
                            if n_sample_triplets % 100 == 0:
                                # print('sampled triplet ids', n_sample_triplets)
                                pass
                            pid_pos = pids[i_pos]
                            pid_neg = self.gen_neg_pid(pids, role)
                            n_sample_triplets += 1
                            task_q.put((pid1, pid_pos, pid_neg))

                            if n_sample_triplets >= self.save_size:
                                for j in range(N_PROC):
                                    task_q.put((None, None, None))
                                return
        for j in range(N_PROC):
            task_q.put((None, None, None))

    def gen_emb_mp(self, task_q, emb_q):
        while True:
            pid1, pid_pos, pid_neg = task_q.get()
            if pid1 is None:
                break
            emb1 = lc.get(pid1)
            emb_pos = lc.get(pid_pos)
            emb_neg = lc.get(pid_neg)
            if emb1 is not None and emb_pos is not None and emb_neg is not None:
                emb_q.put((emb1, emb_pos, emb_neg))
        emb_q.put((False, False, False))

    def gen_triplets_mp(self, role='train'):
        N_PROC = 8

        task_q = mp.Queue(N_PROC * 6)
        emb_q = mp.Queue(1000)

        producer_p = mp.Process(target=self.sample_triplet_ids, args=(task_q, role, N_PROC))
        consumer_ps = [mp.Process(target=self.gen_emb_mp, args=(task_q, emb_q)) for _ in range(N_PROC)]
        producer_p.start()
        [p.start() for p in consumer_ps]

        cnt = 0

        while True:
            if cnt % 1000 == 0:
                print('get', cnt, datetime.now() - start_time)
            emb1, emb_pos, emb_neg = emb_q.get()
            if emb1 is False:
                producer_p.terminate()
                producer_p.join()
                [p.terminate() for p in consumer_ps]
                [p.join() for p in consumer_ps]
                break
            cnt += 1
            yield emb1, emb_pos, emb_neg

    def dump_triplets(self, role='train'):
        triplets = self.gen_triplets_mp(role)
        if role == 'train':
            out_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.save_size))
        else:
            out_dir = join(settings.OUT_DIR, 'test-triplets')
        os.makedirs(out_dir, exist_ok=True)
        anchor_embs = []
        pos_embs = []
        neg_embs = []
        f_idx = 0
        for i, t in enumerate(triplets):
            if i % 100 == 0:
                print(i, datetime.now() - start_time)
            emb_anc, emb_pos, emb_neg = t[0], t[1], t[2]
            anchor_embs.append(emb_anc)
            pos_embs.append(emb_pos)
            neg_embs.append(emb_neg)
            if len(anchor_embs) == self.batch_size:
                data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
                data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
                f_idx += 1
                anchor_embs = []
                pos_embs = []
                neg_embs = []
        if anchor_embs:
            data_utils.dump_data(anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
            data_utils.dump_data(neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        print('dumped')


if __name__ == '__main__':
    data_gen = TripletsGenerator(train_scale=1000000)
    data_gen.dump_triplets(role='train')
    data_gen.dump_triplets(role='test')
