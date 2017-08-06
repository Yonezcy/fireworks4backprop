#encoding=utf-8
#Date 2017.03.10
#三层bp网络
#阀值由每个样本的和共同修正

from Fireworks import *
'''sigmoid函数
参数：
x：自变量，是一个numpy型数组
d：是否进行求导，是一个布尔型变量
返回值：
函数值或者求导的结果，是一个numpy数组
'''
def sigmoid(x, d = False):
	if d == True:
		return x * (1 - x)
	else:
		return 1 / (1 + exp(-x))

'''神经网络训练函数
参数：
X：训练数据集，numpy数组
Y：训练标签集，numpy数组
h：隐层神经元的个数，整型
d：是否使用FWALG算法进行优化，布尔型变量
返回值：
隐层、输出层的权值和阈值，numpy数组
神经网络的迭代误差、烟花算法的迭代误差，整型
'''
def MyBP(X, Y, h, d = False):
	#布尔型变量d表示是否使用烟花算法优化权值和阈值
	if shape(X)[0] != shape(Y)[0]:
		print("The line of X and Y must be same!")

	m = shape(X)[0]; n = shape(X)[1]; l = shape(Y)[1]

	if d == True:
		#输入层-隐层权重
		whj, vih, rh, thetaj, EE = FWA(X, Y, h)
		whj = array(whj); vih = array(vih); rh = array(rh); thetaj = array(thetaj)

	else:
		whj = random.random((n, h))
		vih = random.random((h, l))
		rh = random.random((1, h))
		thetaj = random.random((1, l))
		EE = 0

	a = [0] * 200

	for i in range(200):
		#正向传播
		#隐层输入
		alphah = dot(X, whj)
		#隐层输出
		bh = sigmoid(alphah - rh)
		#输出层输入
		betaj = dot(bh, vih)
		#输出层输出
		ykj = sigmoid(betaj - thetaj)
		#误差
		E = Y - ykj
		a[i] = sum(E * E) / 2

		#反向传播
		gj = sigmoid(ykj, True) * E
		#隐层-输出层权重改变
		delta_vih = dot(bh.T, gj)

		#输出层阀值改变，由所有样本一起修正
		delta_thetaj = -gj[0, :]
		for i in range(1, m):
			delta_thetaj += -gj[i, :]

		#输入层-隐层权重改变
		delta_whj = dot(X.T, ((sigmoid(bh, True)) * dot(gj, vih.T)))

		#隐层阀值改变，由所有样本一起修正
		delta_rh1 = (sigmoid(bh, True)) * dot(gj, vih.T)
		delta_rh = -delta_rh1[0, :]
		for i in range(1, m):
			delta_rh += -delta_rh1[i, :]

		#修正
		vih += delta_vih
		thetaj += delta_thetaj
		whj += delta_whj
		rh += delta_rh

	#返回值为四个参数以及神经网络的迭代误差、烟花算法的迭代误差
	return whj, rh, vih, thetaj, a, EE

'''神经网络预测函数
参数：
X：新样本的数据集，numpy数组
whj：输入层-隐层权值
rh：隐层阈值
vih：隐层-输出层权值
thetaj：输出层阈值
返回值：
Y：新样本的标签，numpy数组
'''
def predictY(X, whj, rh, vih, thetaj):
	alphah = dot(X, whj)
	# python中数组同列不同行可以加减，少行的用上一行补充
	bh = sigmoid(alphah - rh)
	betaj = dot(bh, vih)
	Y = sigmoid(betaj - thetaj)
	return Y