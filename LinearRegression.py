#coding=utf-8
#Date 2017.03.12

from numpy import *
import matplotlib.pyplot as plt

'''线性回归函数
参数：
x：训练集样本集合，numpy数组
y：训练集标签集合，numpy数组
newx：测试集样本集合，numpy数组
newy：测试集标签集合，numpy数组
返回值：
train_error：训练集误差，浮点型
error：测试集误差，浮点型
rr：回归训练集r方
'''
def LinearRegression(x, y, newx, newy):

	xTx = x.T * x
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular")
	ws = xTx.I * (x.T * y)
	yHat = x * ws

	#计算训练集r方
	SSE = (y - yHat).T * (y - yHat)
	error_train = SSE / 2
	#print '训练集均方误差 =',error_train
	yAvg = mean(y)
	SSR = (yHat - yAvg).T * (yHat - yAvg)
	SST = SSR + SSE
	rr = SSR / SST

	#计算测试集上的均方误差
	predict_y = newx * ws
	error = (predict_y - newy).T * (predict_y - newy) / 2
	#print '测试集均方误差 =', error

	#画图
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#画点
	ax.scatter(yHat.flatten().A[0], y.flatten().A[0], color='blue')
	ax.scatter(predict_y.flatten().A[0], newy.flatten().A[0],color='black')
	#画线
	x = [0, 600]
	y = [0, 600]
	ax.plot(x, y, 'r')
	plt.title('linear regression', fontname='times new Roman', fontsize='10.5')
	plt.xlabel('predictvalue', fontname='times new Roman', fontsize='10.5')
	plt.ylabel('realvalue', fontname='times new Roman', fontsize='10.5')
	plt.show()

	return error_train, error, rr