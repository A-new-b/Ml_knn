import numpy as np


# knn 算法
def knn(train_x, t, k):
    dis_list = []
    n = 0
    for i in train_x:
        val = ((i - t) ** 2).sum()
        dic = {'val': val, 'index': n}
        dis_list.append(dic)
        n += 1

    def sort_key(e):
        return e['index']

    dis_list.sort(key=sort_key)
    neighbors = dis_list[0:k]

    return neighbors


class MLKNN:
    s = 1  # 平滑参数
    k = 10  # 默认k值
    label_num = 0
    train_data_num = 0
    train_x = np.array([])  # 向量集
    train_y = np.array([])  # 标签集

    Ph1 = np.array([])
    Ph0 = np.array([])
    Peh1 = np.array([])
    Peh0 = np.array([])

    def __init__(self, train_x, train_y, k, s):
        self.k = k
        self.s = s
        self.train_x = train_x
        self.train_y = train_y
        self.label_num = train_y.shape[1]  # 标签集列数
        self.train_data_num = train_x.shape[0]  # 训练数据数
        self.Ph1 = np.zeros(self.label_num)
        self.Ph0 = np.zeros(self.label_num)
        self.Peh1 = np.zeros([self.label_num, self.k + 1])
        self.Peh0 = np.zeros([self.label_num, self.k + 1])

    def train(self):
        # 先算P(H^1_0)
        for i in range(self.label_num):
            cnt = 0
            for j in range(self.train_data_num):
                if self.train_y[j][i] == 1:
                    cnt = cnt + 1
            self.Ph1[i] = (self.s + cnt) / (self.s * 2 + self.train_data_num)
            self.Ph0[i] = 1 - self.Ph1[i]

        for i in range(self.label_num):  # 对于每一个标签

            print('training for label\n', i)
            c1 = np.zeros(self.k + 1)
            c0 = np.zeros(self.k + 1)

            for j in range(self.train_data_num):
                temp = 0
                neighbors = knn(self.train_x, self.train_x[j], self.k)

                for k in range(self.k):
                    temp = temp + int(self.train_y[int(neighbors[k]['index'])][i])  # 算出每一个实例近邻具有该标签的数量

                if self.train_y[j][i] == 1:  # 若该实例具有该标签
                    c1[temp] = c1[temp] + 1  # 则c1[近邻具有该标签的数量]+1
                else:
                    c0[temp] = c0[temp] + 1

            for j in range(self.k + 1):  # 计算P(E^l_j|H^l_1)
                self.Peh1[i][j] = (self.s + c1[j]) / (self.s * (self.k + 1) + np.sum(c1))  # 这里原代码好像写错了
                self.Peh0[i][j] = (self.s + c0[j]) / (self.s * (self.k + 1) + np.sum(c0))
        np.save('parameter_data/Ph1.npy', self.Ph1)  # 存储
        np.save('parameter_data/Ph0.npy', self.Ph0)
        np.save('parameter_data/Peh1.npy', self.Peh1)
        np.save('parameter_data/Peh0.npy', self.Peh0)

    def test(self):
        test_x = np.load('dataset/test_x.npy', allow_pickle=True)
        test_y = np.load('dataset/test_y.npy', allow_pickle=True)
        predict = np.zeros(test_y.shape, dtype=np.int)
        test_data_num = test_x.shape[0]

        for i in range(test_data_num):
            neighbors = knn(self.train_x, test_x[i], self.k)

            for j in range(self.label_num):
                temp = 0
                for nei in neighbors:
                    temp = temp + int(self.train_y[int(nei['index'])][j])

                if self.Ph1[j] * self.Peh1[j][temp] > self.Ph0[j] * self.Peh0[j][temp]:  # 判断是否有该标签
                    predict[i][j] = 1
                else:
                    predict[i][j] = 0

        np.save('parameter_data/predict.npy', predict)


def HammingLoss(test_y, predict):  # 海明损失
    label_num = test_y.shape[1]
    test_data_num = test_y.shape[0]
    temp = 0
    for i in range(test_data_num):
        temp = temp + np.sum(test_y[i] ^ predict[i])

    hamming_loss = temp / label_num / test_data_num

    return hamming_loss


if __name__ == '__main__':
    k = 10
    s = 1
    train_x = np.load('dataset/train_x.npy', allow_pickle=True)
    train_y = np.load('dataset/train_y.npy', allow_pickle=True)

    mlknn = MLKNN(train_x, train_y, k, s)

    mlknn.train()
    mlknn.test()

    test_y = np.load('dataset/test_y.npy', allow_pickle=True)
    predict = np.load('parameter_data/predict.npy', allow_pickle=True)  # 提取计算好的结果
    test_y = test_y.astype(np.int)

    hamming_loss = HammingLoss(test_y, predict)  # 计算损失
    print('hamming_loss = ', hamming_loss)
