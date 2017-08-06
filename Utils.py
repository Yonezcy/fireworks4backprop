#encoding=utf-8
#Date 2017.5.19
#Fireworks Algorithm

from numpy import  *

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

'''计算适应度函数
烟花个体的解码过程，得到权值和阈值参数，通过这些参数得到适应度即神经网络中的均方误差
参数：
train_x：训练集特征样本，numpy数组
train_y：训练集标签样本，numpy数组
x：当前个体（编码得到），numpy数组
n：输入层神经元的数量，整型
h：隐层神经元的数量，整型
l：输出层神经元的数量，整型
返回值：
适应度函数值（均方误差），浮点型
'''
def calculatef(train_x,train_y,x,n,h,l):
	whj = zeros([n, h])
	vih = zeros([h, l])
	rh = zeros([1, h])
	thetaj = zeros([1, l])

	for i in range(n):
		for j in range(h):
			whj[i][j] = x[i*h+j]

	for i in range(h):
		for j in range(l):
			vih[i][j] = x[n*h+i*l+j]

	for i in range(h):
		rh[0][i] = x[n*h+h*l+i]

	for i in range(l):
		thetaj[0][i] = x[n*h+h*l+h+i]

	#计算适应度函数（均方误差）
	# 隐层输入
	alphah = dot(train_x, whj)
	# 隐层输出
	bh = sigmoid(alphah - rh)
	# 输出层输入
	betaj = dot(bh, vih)
	# 输出层输出
	ykj = sigmoid(betaj - thetaj)
	# 误差
	E = train_y - ykj
	#均方误差
	E = sum(E*E)/2
	return E

'''解码函数
参数：
x：当前个体（编码得到），numpy数组
n：输入层神经元的数量，整型
h：隐层神经元的数量，整型
l：输出层神经元的数量，整型
E：均方误差
返回值：
输入层-隐层的权重，隐层阈值，隐层-输出层权重，输出层阈值，均为numpy数组
E：均方误差
'''
def final_weight(x, n, h, l, E):
	whj = zeros([n, h])
	vih = zeros([h, l])
	rh = zeros([1, h])
	thetaj = zeros([1, l])

	for i in range(n):
		for j in range(h):
			whj[i][j] = x[i*h+j]

	for i in range(h):
		for j in range(l):
			vih[i][j] = x[n*h+i*l+j]

	for i in range(h):
		rh[0][i] = x[n*h+h*l+i]

	for i in range(l):
		thetaj[0][i] = x[n*h+h*l+h+i]

	return whj, vih, rh, thetaj, E