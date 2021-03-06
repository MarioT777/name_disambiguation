import os
from os.path import join

import numpy as np
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model, model_from_json
from keras.optimizers import Adam

from global_.embedding import EMB_DIM
from global_.triplet import l2Norm, euclidean_distance, triplet_loss, accuracy
from utils import data_utils
from utils import settings

"""
全局度量学习模型
global metric learning model
"""


class GlobalTripletModel:

    def __init__(self, data_scale):
        self.data_scale = data_scale
        self.train_triplets_dir = join(settings.OUT_DIR, 'triplets-{}'.format(self.data_scale))  # 训练集目录
        self.test_triplets_dir = join(settings.OUT_DIR, 'test-triplets')  # 测试集目录
        self.train_triplet_files_num = self.get_triplets_files_num(self.train_triplets_dir)  # 训练集文件数
        self.test_triplet_files_num = self.get_triplets_files_num(self.test_triplets_dir)  # 测试集文件数
        print('test file num', self.test_triplet_files_num)

    @staticmethod
    def get_triplets_files_num(path_dir):
        """
        计算文件数
        :param path_dir:
        :return:
        """
        files = []
        for f in os.listdir(path_dir):
            if f.startswith('anchor_embs_'):
                files.append(f)
        return len(files)

    def load_batch_triplets(self, f_idx, role='train'):
        if role == 'train':
            cur_dir = self.train_triplets_dir
        else:
            cur_dir = self.test_triplets_dir
        X1 = data_utils.load_data(cur_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
        X2 = data_utils.load_data(cur_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
        X3 = data_utils.load_data(cur_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        return X1, X2, X3

    def load_triplets_data(self, role='train'):
        """
        加载三元组数据
        :param role:
        :return:
        """
        X1 = np.empty([0, EMB_DIM])  # 创建数组
        X2 = np.empty([0, EMB_DIM])
        X3 = np.empty([0, EMB_DIM])
        if role == 'train':  # 取出对应文件数目
            f_num = self.train_triplet_files_num
        else:
            f_num = self.test_triplet_files_num
        for i in range(f_num):
            print('load', i)
            x1_batch, x2_batch, x3_batch = self.load_batch_triplets(i, role)  # 加载相应数据
            p = np.random.permutation(len(x1_batch))  # 生成长度为X1_batch的乱序数组
            x1_batch = np.array(x1_batch)[p]  # 将anchor样本集打乱
            x2_batch = np.array(x2_batch)[p]
            x3_batch = np.array(x3_batch)[p]
            X1 = np.concatenate((X1, x1_batch))  # 数组拼接
            X2 = np.concatenate((X2, x2_batch))
            X3 = np.concatenate((X3, x3_batch))
        return X1, X2, X3

    @staticmethod
    def create_triplet_model():
        """
        构建三元组模型
        :return:
        """
        emb_anchor = Input(shape=(EMB_DIM, ), name='anchor_input')  # 返回维数为EMB_DIM=100的一维数组
        emb_pos = Input(shape=(EMB_DIM, ), name='pos_input')
        emb_neg = Input(shape=(EMB_DIM, ), name='neg_input')

        # shared layers
        # Dense 全连接层，第一个参数表示输出的维度，即输出的是一个N*(输出的维度)的矩阵
        layer1 = Dense(128, activation='relu', name='first_emb_layer')  # 第一层,设定输出维度，激活函数，层名
        layer2 = Dense(64, activation='relu', name='last_emb_layer')  # 第二层
        # 归一化
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[64])

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))  # 全连层(128) -> 全连层(64) -> l2标准化层(64)
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))  # 权重共享
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))

        pos_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])  # 欧几里得距离
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            """
            根据输入张量的维度，计算输出张量的维度
            :param input_shape:
            :return:
            """
            shape = list(input_shape[0])  # input_shape[0]表示输入张量的维度
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2  # 最后一维 *2
            return tuple(shape)  # 将列表转换为元组

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),  # K.stack 将一个列表中维度数目为R的张量堆积起来形成维度为R+1的新张量
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        # 整个模型(anchor, pos, neg) -> (pos_dis, neg_dis)
        model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])
        # 编译模型，损失函数，优化器，评估标准函数

        inter_layer = Model(inputs=model.get_input_at(0), outputs=model.get_layer('norm_layer').get_output_at(0))
        #  中间层anchor -> l2Norm(y)

        return model, inter_layer

    def load_triplets_model(self):
        """
        加载三元组模型
        :return:
        """
        model_dir = join(settings.OUT_DIR, 'model')  # 设定模型目录
        rf = open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'r')
        model_json = rf.read()
        rf.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))  # 加载模型权重
        return loaded_model

    def train_triplets_model(self):
        """
        训练三元组模型
        :return:
        """
        X1, X2, X3 = self.load_triplets_data()  # 加载训练集数据
        n_triplets = len(X1)  # 总共三元组数量
        print('loaded')
        model, inter_layer = self.create_triplet_model()
        # model:(anchor, pos, neg)->(pos_dis, neg_dis)，inter_model:(anchor->l2Norm(y))
        # print(model.summary())

        X_anchor, X_pos, X_neg = X1, X2, X3
        X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}  # 建立字典，作为网络输入

        # 训练模型数据，标记=n_triplets*2 的纯1数组， 窗口大小， 训练次数， 是否混洗， 交叉验证率
        model.fit(X, np.ones((n_triplets, 2)), batch_size=64, epochs=5, shuffle=True, validation_split=0.2)

        model_json = model.to_json()  # 将模型保存为json
        model_dir = join(settings.OUT_DIR, 'model')
        os.makedirs(model_dir, exist_ok=True)
        with open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'w') as wf:  # 创建文件并保存
            wf.write(model_json)
        model.save_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))  # 保存模型的权重

        # test_triplets = self.load_triplets_data(role='test')  # 加载三元组测试集数据
        # auc_score = eval_utils.full_auc(model, test_triplets)  # 评估模型，返回AUC分数
        # # print('AUC', auc_score)
        #
        # loaded_model = self.load_triplets_model()  # 加载模型
        # print('triplets model loaded')
        # auc_score = eval_utils.full_auc(loaded_model, test_triplets)

    # def evaluate_triplet_model(self):
    #     test_triplets = self.load_triplets_data(role='test')
    #     loaded_model = self.load_triplets_model()
    #     print('triplets model loaded')
    #     auc_score = eval_utils.full_auc(loaded_model, test_triplets)


if __name__ == '__main__':
    global_model = GlobalTripletModel(data_scale=1000000)
    global_model.train_triplets_model()
    print('done')