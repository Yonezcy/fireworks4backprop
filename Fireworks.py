#encoding=utf-8
#Date 2017.5.19
#Fireworks Algorithm

from Utils import *

'''烟花算法
目的是通过烟花算法得到较好的神经网络参数的初始值，再使用梯度下降进行迭代
参数：
X：训练集样本集合，numpy数组
Y：训练节标签集合，numpy数组
h：隐层神经元数量
返回值：
优化后的输入层-隐层的权重，隐层阈值，隐层-输出层权重，输出层阈值初始值，均为numpy数组
'''
def FWA(X, Y, h):
	if shape(X)[0] != shape(Y)[0]:
		print("The line of X and Y must be same!")
	m = shape(X)[0]; n = shape(X)[1]; l = shape(Y)[1]

	#初始化算法参数
		#火花总数
	m = 50
		#上限和下限
	a = 0.8; b = 0.04
		#爆炸幅度
	A = 40
		#烟花高斯爆炸个数
	mm = 5
		#烟花维度
	dimension = n*h + h*l + h + l
		#烟花个数
	nn = 5
		#最大、最小边界
	xmin = -5; xmax = 5

	#初始化烟花
	fireworks = zeros([nn, dimension])
	for i in range(nn):
		for j in range(dimension):
			fireworks[i][j] = random.uniform(-5, 5)

	#初始化新烟花
	fireworks_new = zeros([nn, 100, dimension])

	#初始化高斯火花
	fireworks_rbf = zeros([nn, dimension])

	#产生火花
		#每个烟花产生火花的数量
	Si = zeros([nn, 1])
		#每个烟花的爆炸半径
	Ai = zeros([nn, 1])
		#火花限制
	si = zeros([nn, 1])
		#计算每个烟花的适应度函数值
	f = zeros([nn, 1])
		#最大最小适应度
	fmax = f[0]; fmin = f[nn-1]
		#误差函数初始化
	E = zeros([5000, 1])


	#烟花算法迭代过程
	for delta_num in range(5000):

		# 普通爆炸产生的火花总数
		sum_new_fireworks = 0
		# 总适应度
		sum = 0
		#计算适应度,并求出最大值和最小值
		for i in range(nn):
			f[i] = calculatef(X, Y, fireworks[i], n, h, l)
			if f[i] > fmax:
				fmax = f[i]
			if f[i] < fmin:
				fmin = f[i]
			sum += f[i]

			#求每个烟花的爆炸半径和火花数
		for i in range(nn):
				#计算火花数
			Si[i] = m * (fmax - f[i] + 0.0001) / (nn * fmax - sum + 0.0001)
			Si[i] = round(Si[i])
			if Si[i] < a * m:
				si[i] = round(a * m)
			elif Si[i] > b * m:
				si[i] = round(b * m)
			else:
				si[i] = round(Si[i])
				#不能超过火花数限制
			if Si[i] > si[i]:
				Si[i] = si[i]

				#计算普通爆炸产生的火花总数
			sum_new_fireworks += int(Si[i])

			#计算爆炸半径
			Ai[i] = A * (f[i] - fmin + 0.0001) / (sum - nn * fmin + 0.0001)

				#产生新火花
			for j in range(Si[i]):
					#初始化新火花
				fireworks_new[i][j] = fireworks[i]
					#随机选择z个维度
				z = random.randint(1, dimension)
					#打乱随机选择前z个
				zz = range(dimension)
				random.shuffle(zz)

					# 产生新火花
				for k in range(z):
					fireworks_new[i][j][zz[k]] += random.uniform(0, Ai[i])


		#产生高斯火花（每个烟花产生一个高斯火花）
			# 随机选择z个维度
		z = random.randint(1, dimension)
		zz = range(dimension)
		random.shuffle(zz)
			#高斯随机数
		g = random.uniform(-1, 1)
			#高斯爆炸算子
		for i in range(mm):
			for j in range(z):
				fireworks_rbf[i][zz[j]] = g * fireworks[i][zz[j]]


			#构造总烟花
		sum_fireworks = nn + sum_new_fireworks + mm
		fireworks_final = zeros([sum_fireworks, dimension])
		for i in range(nn):
			fireworks_final[i] = fireworks[i]

		for j in range(Si[0]):
			fireworks_final[nn+j] = fireworks_new[0][j]

		for i in range(nn-1):
			for j in range(Si[i+1]):
				#print 'Si = ',Si[i]
				fireworks_final[int(nn+j+Si[i])] = fireworks_new[i+1][j]

		for i in range(mm):
			fireworks_final[int(nn+sum_new_fireworks+i)] = fireworks_rbf[i]


		#映射规则
		for i in range(sum_fireworks):
			for j in range(dimension):
				if fireworks_final[i][j] > xmax or fireworks_final[i][j] < xmin:
					fireworks_final[i][j] = xmin + mod(abs(fireworks_final[i][j]), \
				                           (xmax - xmin))

		# 选择策略
			#爆炸后新种群适应度
		f_new = zeros([sum_fireworks, 1])
		f_new_min = f_new[0]
		#print f_new_min
			#初始化最优适应度下标
		min_i = 0
			#选出下一代nn个个体，由最大适应度个体与距离更远的nn-1个个体组成
			#求最优适应度
		for i in range(sum_fireworks):
			#print fireworks_final[i]
			f_new[i] = calculatef(X, Y, fireworks_final[i], n, h, l)
			if f_new[i] < f_new_min:
				f_new_min = f_new[i]
				min_i = i


			#求出每个个体被选择的概率
			#初始化两两个体之间的距离
		D = zeros([sum_fireworks, sum_fireworks])
			#计算两两个体之间的距离
		for i in range(sum_fireworks):
			for j in range(sum_fireworks):
				D[i][j] = dot((fireworks_final[i] - fireworks_final[j]), \
				              (fireworks_final[i] - fireworks_final[j])) / 2

			#初始化每个个体与其他个体之间的距离之和
		Ri = zeros([sum_fireworks, 1])
			#初始化距离矩阵的副本
		RRi = zeros([sum_fireworks, 1])
			#计算每个个体与其他个体的距离之和
		for i in range(sum_fireworks):
			for j in range(sum_fireworks):
				Ri[i] += D[i][j]
		RRi = Ri

			#选出距离最远的nn-1个个体，即对距离矩阵进行排序
		for i in range(sum_fireworks-1):
			for j in range(i, sum_fireworks):
				if Ri[i] < Ri[j]:
					tmp = Ri[i]
					Ri[i] = Ri[j]
					Ri[j] = tmp

			#构造新种群
		fireworks[0] = fireworks_final[min_i]
		for i in range(sum_fireworks):
			if Ri[0] == RRi[i]:
				fireworks[1] = fireworks_final[i]
			if Ri[1] == RRi[i]:
				fireworks[2] = fireworks_final[i]
			if Ri[2] == RRi[i]:
				fireworks[3] = fireworks_final[i]
			if Ri[3] == RRi[i]:
				fireworks[4] = fireworks_final[i]

			#迭代完毕，返回最优个体
			#初始化最优适应度下标
		ii = 0
		for i in range(nn):
			f[i] = calculatef(X, Y, fireworks[i], n, h, l)
			fmin = f[0]
			if f[i] < fmin:
				ii = i

		E[delta_num] = f[ii]

	return final_weight(fireworks[ii], n, h, l, E)