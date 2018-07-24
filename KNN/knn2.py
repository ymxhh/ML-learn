# coding: utf-8
from sklearn.datasets import load_iris
from sklearn import neighbors
import sklearn

iris = load_iris()
print(iris)

knn = neighbors.KNeighborsClassifier()

# 训练数据集
knn.fit(iris.data, iris.target)

# 训练准确率
score = knn.score(iris.data, iris.target)

# 预测
predict = knn.predict([[0.1, 0.2, 0.3, 0.4]])
# 预测，返回概率数组
predict2 = knn.predict_proba([[0.1, 0.2, 0.3, 0.4]])

print(predict)
print(iris.target_names[predict])