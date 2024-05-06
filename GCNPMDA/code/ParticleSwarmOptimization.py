import numpy as np
from main import cross_validation_experiment

# 定义粒子群优化算法类，参考了网上的一些代码
class ParticleSwarmOptimization():
    def __init__(self, epoch, criterion, circle_time,
                    disease_microbe_matrix, disease_sim,
                    microbe_sim,seed, epch, adjdp,
                    bestAUC,swarm_size=20, max_iter=100, w=0.9, c1=2.05, c2=2.05):
        # 初始化模型、数据加载器、损失函数等属性
        #         self.model = model
        self.epoch = epoch
        #         self.data_loader = data_loader
        self.criterion = criterion
        self.circle_time = circle_time
        self.disease_microbe_matrix = disease_microbe_matrix
        self.disease_sim = disease_sim
        self.microbe_sim = microbe_sim
        self.seed = seed
        self.epoch = epch
        self.adjdp = adjdp
        self.bestAUC = bestAUC

        # 初始化粒子群的大小、最大迭代次数、惯性权重、学习因子等参数
        # 你可以根据需要修改这些参数
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        """"后期需要自己增加参数，记得数总参数数据也要修改，self.n_params=6"""
        # 初始化粒子群中每个粒子的位置、速度、适应度值、最佳位置和最佳适应度值等属性
        # 这里假设要优化的参数有6个：num_filters_1, num_filters_2, kernel_size_1, kernel_size_2, num_neurons 和 activation
        # 每个参数都有一个取值范围和一个步长，用于限制搜索空间和离散化搜索空间
        # 继续定义粒子群优化算法类

        self.n_params = 4  ##可以把下面的6替换成self.n_params

        self.position = np.zeros((self.swarm_size, self.n_params))  # 位置矩阵，每行表示一个粒子的位置，每列表示一个参数的值
        self.velocity = np.zeros((self.swarm_size, self.n_params))  # 速度矩阵，每行表示一个粒子的速度，每列表示一个参数的变化量
        self.fitness = np.zeros(self.swarm_size)  # 适应度值向量，每个元素表示一个粒子的适应度值
        self.pbest_position = np.zeros((self.swarm_size, self.n_params))  # 最佳位置矩阵，每行表示一个粒子的最佳位置，每列表示一个参数的值
        self.pbest_fitness = np.full(self.swarm_size, float('inf'))  # 最佳适应度值向量，每个元素表示一个粒子的最佳适应度值
        self.gbest_position = np.zeros(self.n_params)  # 全局最佳位置向量，每个元素表示一个参数的值
        self.gbest_fitness = 99  # float('inf') # 全局最佳适应度值标量
        # lb = [10, 0.0, 0.0001,1]  # emb_dim至少为10，dp和lr,CNNlayer的下界
        # ub = [260, 0.1, 0.1,100000]  # emb_dim的上界为100，dp和lr,CNNlayer的上界
        self.ranges = [(10, 100), (0.0, 1), (0.0001, 0.1), (1, 10)]
        self.steps = [10,0.1, 0.01,1]
        self.bestmodelname = ""

    def evaluate(self):
        # 随机初始化粒子群中每个粒子的位置和速度，并计算其适应度值和最佳位置和最佳适应度值
        for i in range(self.swarm_size):
            for j in range(self.n_params):
                if self.steps[j] is None:  # 如果参数是离散型，则随机选择其中一个可能取值
                    self.position[i, j] = np.random.choice(self.ranges[j])
                else:  # 如果参数是连续型，则在取值范围内随机生成一个可能取值，并按照步长进行四舍五入
                    self.position[i, j] = round(
                        np.random.uniform(self.ranges[j][0], self.ranges[j][1]) / self.steps[j]) * self.steps[j]
            self.velocity[i, :] = np.random.uniform(-1, 1, self.n_params)  # 在[-1,1]之间随机生成一个速度向量
            self.fitness[i] = self.objective_function(self.position[i, :], self.circle_time,
                    self.disease_microbe_matrix,
                    self.disease_sim,
                    self.microbe_sim,
                    self.seed,
                    self.epoch,
                    self.adjdp,
                    self.bestAUC)  # 计算当前位置的适应度值
            print("position=")
            print(self.position[i, :])
            print("fitness=")
            print(+self.fitness[i])
            # print("modelname=")
            # print(modelname)
            self.pbest_position[i, :] = self.position[i, :]  # 初始化最佳位置为当前位置
            self.pbest_fitness[i] = self.fitness[i]  # 初始化最佳适应度值为当前适应度值
            if self.fitness[i] < self.gbest_fitness:  # 如果当前适应度值小于全局最佳适应度值，则更新全局最佳位置和全局最佳适应度值
                self.gbest_position = self.position[i, :]
                self.gbest_fitness = self.fitness[i]
                # self.bestmodelname = modelname
                print("gbest_position=")
                print(self.gbest_position)
                print("gbest_fitness=")
                print(+self.gbest_fitness)
                print("bestmodelname=")
                # print(self.bestmodelname)

    # 定义评估函数，用于计算给定位置的适应度值
    # 这里使用均方根误差（RMSE）作为损失函数，因此适应度值越小越好
    def objective_function(x, circle_time, disease_microbe_matrix, disease_sim, microbe_sim, seed, epoch, adjdp,
                           bestAUC):
        # emb_dim, dp, lr, CNNLayer = int(x[0]), x[1], x[2],int(x[3])
        emb_dim, dp, lr, CNNLayer = 256, 0.0005, 0.01, 2
        print("emb_dim, dp, lr, CNNLayer", emb_dim, dp, lr, CNNLayer)
        result = np.zeros((circle_time, 7), float)
        for i in range(circle_time):
            result[i, :] = cross_validation_experiment(
                disease_microbe_matrix,
                disease_sim,
                microbe_sim,
                seed,
                epoch,
                emb_dim,
                dp,
                lr,
                adjdp,
                bestAUC, CNNLayer)
        column_means = np.mean(result, axis=0)
        # if column_means[1]>0.88 and column_means[1]>0.92:
        #     return 1-np.mean(column_means[1])
        #     # 假设我们最小化结果的平均值
        # else:
        #     return 1
        # return (1-np.mean(column_means[1]))*(1-np.mean(column_means[0]))
        return 1 - np.mean(column_means[0])



    def update(self):
        # 更新方法，用来根据公式更新每个粒子的速度和位置

        r1 = np.random.uniform(size=(self.swarm_size, self.n_params))
        r2 = np.random.uniform(size=(self.swarm_size, self.n_params))

        # 根据公式更新速度和位置矩阵
        velocity = self.w * self.velocity + self.c1 * r1 * (self.pbest_position - self.position) + self.c2 * r2 * (
                    self.gbest_position - self.position)
        position = self.position + self.velocity

    def optimize(self):
        # 优化方法，用来执行给定次数的迭代，并输出最佳参数和适应度值

        print("开始PSO优化...")

        for iter in range(self.max_iter):
            print("第{}次迭代：".format(iter + 1))

            self.evaluate()

            print("当前全局最佳适应度值：{:.4f}".format(self.gbest_fitness))
            print("当前全局最佳参数：")
            print(self.gbest_position)

            self.update()

        print("PSO优化结束！")
        print("最终全局最佳适应度:")
        print(self.gbest_position)
        return self.gbest_position, self.bestmodelname