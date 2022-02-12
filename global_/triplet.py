from keras import backend as K


def l2Norm(x):  # l2_normalization标准化，x_i = x_i/norm(x) norm(x)=sqrt(sum(x)^2)
    return K.l2_normalize(x, axis=-1)  # 基于L2范数进行标准化，各维度变换到0·1之间


def euclidean_distance(vects):  # 计算欧几里得距离
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
    # 返回L2范数即欧式距离，其中epsilon表示很小的常数（le-7）


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0, 0]) - K.square(y_pred[:, 1, 0]) + margin))


def accuracy(_, y_pred):
    return K.mean(y_pred[:, 0, 0] < y_pred[:, 1, 0])
