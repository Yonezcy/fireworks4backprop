#coding=utf-8
#Date 2017.03.12

from Bp_train import *
from numpy import *
import matplotlib.pyplot as plt

'''神经网络调用函数
参数：
x：训练集样本集合，numpy数组
y：训练集标签集合，numpy数组
newx：测试集样本集合，numpy数组
newy：测试集标签集合，numpy数组
d：布尔型变量，是否使用fw算法进行优化
返回值：
train_error：训练集误差，浮点型
error：测试集误差，浮点型
'''
def BPneuralnetwork(x, y, newx, newy, d):
	#训练集归一化
	xx = zeros([shape(x)[0], shape(x)[1]])
	yy = zeros([shape(y)[0], shape(y)[1]])

	for i in range(shape(x)[0]):
		for j in range(shape(x)[1]):
			xx[i, j] = (x[i, j] - min(x[:, j])) / (max(x[:, j] - min(x[:, j])))

	for i in range(shape(y)[0]):
		for j in range(shape(y)[1]):
			yy[i, j] = (y[i, j] - min(y[:, j])) / (max(y[:, j] - min(y[:, j])))

	#求训练集均方误差
	#a为神经网络的迭代误差、EE为烟花算法的迭代误差
	whj, rh, vih, thetaj, a, EE = MyBP(xx, yy, 6, d)
	alphah = dot(xx, whj)
	bh = sigmoid(alphah - rh)
	betaj = dot(bh, vih)
	NewY = sigmoid(betaj - thetaj)
	NewYY = zeros([shape(NewY)[0], shape(NewY)[1]])
	for i in range(shape(NewY)[0]):
		for j in range(shape(NewY)[1]):
			NewYY[i, j] = NewY[i, j] * (max(y[:, j] - min(y[:, j]))) + \
			                  min(y[:, j])

	train_error = sum((NewYY - y) * (NewYY - y)) / 2

	#测试集
	newxx = zeros([shape(newx)[0], shape(newx)[1]])
	#测试集归一化
	for i in range(shape(newx)[0]):
		for j in range(shape(newx)[1]):
			newxx[i, j] = (newx[i, j] - min(x[:, j])) / (max(x[:, j] - min(x[:, j])))

	#神经网络预测
	predict_y = predictY(newxx, whj, rh, vih, thetaj)
	#预测值反归一化
	newpredict_y = zeros([shape(predict_y)[0], shape(predict_y)[1]])
	for i in range(shape(predict_y)[0]):
		for j in range(shape(predict_y)[1]):
			newpredict_y[i, j] = predict_y[i, j] * (max(y[:, j] - min(y[:, j]))) + \
			                  min(y[:, j])

	#求测试集均方误差
	error = sum((newpredict_y - newy) * (newpredict_y - newy)) / 2

	#画图
	#真实值与预测值图像
	fig1 = plt.figure()
	ax = fig1.add_subplot(111)
	#画点
	ax.scatter(NewYY.flatten(), y.flatten(), color = 'blue')
	ax.scatter(newpredict_y.flatten(), newy.flatten(), color = 'black')
	#画线
	x = [0, 600]
	y = [0, 600]
	ax.plot(x, y, 'r')
	plt.title('BP neural network', fontname='times new Roman', fontsize='10.5')
	plt.xlabel('predictvalue', fontname='times new Roman', fontsize='10.5')
	plt.ylabel('realvalue', fontname='times new Roman', fontsize='10.5')
	plt.show()

	#神经网络误差函数图像
	fig2 = plt.figure()
	bx = fig2.add_subplot(111)
	x = range(200)
	y = a
	bx.plot(x, y)
	plt.title('Error function', fontname='times new Roman', fontsize='10.5')
	plt.xlabel('Number of iterations', fontname='times new Roman', fontsize='10.5')
	plt.ylabel('Error', fontname='times new Roman', fontsize='10.5')
	plt.show()

	return train_error, error