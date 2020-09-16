# Ml_KNN
## 参考
算法层面上参考了https://github.com/hinanmu/MLKNN的MLKNN算法，
感谢大佬，个人修改了部分knn算法和计算过程中出错的部分
<br>
训练集和测试集内容来自http://mulan.sourceforge.net/datasets-mlc.html
使用了birds的训练集。
## 运行
如果要运行已经训练好的birds集合，可以将main.py中的<br>
`mlknn.train()`<br>
` mlknn.test()`<br>
注释掉直接运行，可以看见最后的hamming_loss =  0.05100211829884308 <br>
如果要使用新数据集，那请去上文中的网上下新的arff文件，修改data_loading.py并运行后运行main.py
## 下载
如果不没有接触过github的同学可以点击右上角的code按钮，然后点击download zip即可下载



