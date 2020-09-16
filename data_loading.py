import pandas as pd
import numpy as np
from scipy.io import arff

train_file_path = 'birds/birds-train.arff'
test_file_path = 'birds/birds-test.arff'  # 从“http://mulan.sourceforge.net/datasets-mlc.html”下载数据集

train_data, train_meta = arff.loadarff(train_file_path)
test_data, test_meta = arff.loadarff(test_file_path)  # 从arff文件中导入训练，测试集

# print(len(train_data))
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_data = train_df.values
test_data = test_df.values

print(train_df.shape[0])
print(train_data.shape[0])
# print(train_df.shape[1])train_x = np.array(train_data[:,0:257])

# 将数据集中的向量和标签分开
# 因为鸟类有两个nominal属性位于258,259,所以要去掉
train_x = np.array(train_data[:, 0:258])
train_y = np.array(train_data[:, 260:])
test_x = np.array(test_data[:, 0:258])
test_y = np.array(test_data[:, 260:])

# print(train_y[-40])
# print(train_x[-40])

np.save('dataset/train_x.npy', train_x)
np.save('dataset/train_y.npy', train_y)
np.save('dataset/test_x.npy', test_x)
np.save('dataset/test_y.npy', test_y)  # 保存
