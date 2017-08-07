#coding=utf-8
#Date 2017.06.05

from LinearRegression import *
from BPnn import *
from NonLinearRegression import *
from BP_NonLinearRegression import *

'''从本地加载数据集
参数：
fileName：数据集名称，字符串，若不在当前路径下应为绝对路径
返回值：
dataMat：训练集特征集合，list型
labelMat：训练集标签集合，list型
'''
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), \
                        float(lineArr[3]), float(lineArr[4])])
        labelMat.append(float(lineArr[5]))
    return dataMat, labelMat

print '请选择需要使用的建模方法：'
print '1:线性回归\n','2:非线性回归\n','3:神经网络\n','4:非线性回归+神经网络\n',\
	  '5:烟花算法优化神经网络\n'
a = raw_input("Enter your input: ")
if a == '1':
	#n次k折交叉验证
	dataMat, labelMat = loadDataSet('9.txt')
	dataMat = mat(dataMat); labelMat = mat(labelMat)
	labelMat = labelMat.T
	sum_error_train = 0; sum_error = 0; sum_rr = 0
	for j in range(10):
		z = range(27)
		random.shuffle(z)
		x = zeros([20, 5]); y = zeros([20, 1]); newx = zeros([7, 5]); newy = zeros([7, 1])
		x = mat(x); y = mat(y); newx = mat(newx); newy = mat(newy)
		for i in range(20):
			x[i] = dataMat[z[i]]
			y[i] = labelMat[z[i]]
		for i in range(20, 27):
			newx[i-20] = dataMat[z[i]]
			newy[i-20] = labelMat[z[i]]
		# 调用线性回归方法，得到训练集和测试集误差和r方
		error_train, error, rr = LinearRegression(x, y, newx, newy)
		sum_error_train += error_train; sum_error += error; sum_rr += rr
	error_train = sum_error_train / 10; error = sum_error / 10; rr = sum_rr / 10
	print '训练集均方误差=', error_train
	print '验证集均方误差=', error
	print 'r方=', rr

elif a == '2':
	#n次k折交叉验证
	dataMat, labelMat = loadDataSet('10.txt')
	dataMat = mat(dataMat); labelMat = mat(labelMat)
	labelMat = labelMat.T
	sum_error_train = 0; sum_error = 0; sum_rr = 0
	for j in range(10):
		z = range(27)
		random.shuffle(z)
		x = zeros([20, 5]); y = zeros([20, 1]); newx = zeros([7, 5]); newy = zeros([7, 1])
		x = mat(x); y = mat(y); newx = mat(newx); newy = mat(newy)
		for i in range(20):
			x[i] = dataMat[z[i]]
			y[i] = labelMat[z[i]]
		for i in range(20, 27):
			newx[i-20] = dataMat[z[i]]
			newy[i-20] = labelMat[z[i]]
		# 调用非线性回归方法，得到训练集和测试集误差和r方
		error_train, error, rr = NonLinearRegression(x, y, newx, newy)
		sum_error_train += error_train; sum_error += error; sum_rr += rr
	error_train = sum_error_train / 10; error = sum_error / 10; rr = sum_rr / 10
	print '训练集均方误差=', error_train
	print '验证集均方误差=', error
	print 'r方=', rr

elif a == '3':
	x, y = loadDataSet('1.txt')
	newx, newy = loadDataSet('2.txt')
	x = array(x); y = array([y]); newx = array(newx); newy = array([newy])
	y = y.T; newy = newy.T
	xx = zeros([shape(x)[0], shape(x)[1]-1])
	newxx = zeros([shape(newx)[0], shape(newx)[1]-1])
	for i in range(shape(x)[0]):
		for j in range(1, shape(x)[1]):
			xx[i][j-1] = x[i][j]
	for i in range(shape(newx)[0]):
		for j in range(1, shape(newx)[1]):
			newxx[i][j-1] = newx[i][j]
	# 调用神经网络方法，得到训练集和测试集误差
	train_error, error = BPneuralnetwork(xx, y, newxx, newy, False)
	print '训练集均方误差=', train_error
	print '验证集均方误差=', error

elif a == '4':
	x, y = loadDataSet('3.txt')
	newx, newy = loadDataSet('4.txt')
	x = array(x); y = array([y]); newx = array(newx); newy = array([newy])
	y = y.T; newy = newy.T
	xx = zeros([shape(x)[0], shape(x)[1]-1])
	newxx = zeros([shape(newx)[0], shape(newx)[1]-1])
	for i in range(shape(x)[0]):
		for j in range(1, shape(x)[1]):
			xx[i][j-1] = x[i][j]
	for i in range(shape(newx)[0]):
		for j in range(1, shape(newx)[1]):
			newxx[i][j-1] = newx[i][j]
	# 调用神经网络与非线性回归结合方法（不使用智能优化算法），得到训练集和测试集误差
	train_error, error = BP_NonLinearRegression(xx, y, newxx, newy, False)
	print '训练集均方误差=', train_error
	print '验证集均方误差=', error

else:
	x, y = loadDataSet('3.txt')
	newx, newy = loadDataSet('4.txt')
	x = array(x); y = array([y]); newx = array(newx); newy = array([newy])
	y = y.T; newy = newy.T
	xx = zeros([shape(x)[0], shape(x)[1]-1])
	newxx = zeros([shape(newx)[0], shape(newx)[1]-1])
	for i in range(shape(x)[0]):
		for j in range(1, shape(x)[1]):
			xx[i][j-1] = x[i][j]
	for i in range(shape(newx)[0]):
		for j in range(1, shape(newx)[1]):
			newxx[i][j-1] = newx[i][j]
	# 调用神经网络与非线性回归结合方法（使用智能优化算法），得到训练集和测试集误差
	train_error, error = BP_NonLinearRegression(xx, y, newxx, newy, True)
	print '训练集均方误差=', train_error
	print '验证集均方误差=', error
