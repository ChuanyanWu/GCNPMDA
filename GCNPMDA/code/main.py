import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import gc
import random


from clac_metric import cv_model_evaluate
from utils import *
from model import GCNModel
from opt import Optimizer
from tensorflow.python.client import device_lib
import os
import pickle
# from pyswarm import pso
# from functools import partial
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# 定义粒子群优化算法类，参考了网上的一些代码
class ParticleSwarmOptimization():
    def __init__(self,  circle_time,
                    disease_microbe_matrix, disease_sim,
                    microbe_sim,seed, epch, adjdp,
                    bestAUC,swarm_size=20, max_iter=100, w=0.9, c1=2.05, c2=2.05):
        # 初始化模型、数据加载器、损失函数等属性
        #         self.model = model
        # self.epoch = epoch
        #         self.data_loader = data_loader
        # self.criterion = criterion
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

        self.n_params = 2  ##可以把下面的6替换成self.n_params

        self.position = [250, 0.0005]# np.zeros((self.swarm_size, self.n_params))  # 位置矩阵，每行表示一个粒子的位置，每列表示一个参数的值
        self.velocity = [10, 0.0005]#np.zeros((self.swarm_size, self.n_params))  # 速度矩阵，每行表示一个粒子的速度，每列表示一个参数的变化量
        self.fitness = np.zeros(self.swarm_size)  # 适应度值向量，每个元素表示一个粒子的适应度值
        self.pbest_position = np.zeros((self.swarm_size, self.n_params))  # 最佳位置矩阵，每行表示一个粒子的最佳位置，每列表示一个参数的值
        self.pbest_fitness = np.full(self.swarm_size, float('inf'))  # 最佳适应度值向量，每个元素表示一个粒子的最佳适应度值
        self.gbest_position = np.zeros(self.n_params)  # 全局最佳位置向量，每个元素表示一个参数的值
        self.gbest_fitness = 99  # float('inf') # 全局最佳适应度值标量
        # lb = [10, 0.0, 0.0001,1]  # emb_dim至少为10，dp和lr,CNNlayer的下界
        # ub = [260, 0.1, 0.1,100000]  # emb_dim的上界为100，dp和lr,CNNlayer的上界
        self.ranges = [(250, 300), (0.01, 1)]#, (0.0001, 0.1), (2, 10)]
        self.steps = [10,0.1]#, 0.01]#,1]
        self.bestmodelname = ""
        # print("1 pso init")

    def evaluate(self):
        # print("3 pso evaluate")
        # 随机初始化粒子群中每个粒子的位置和速度，并计算其适应度值和最佳位置和最佳适应度值
        for i in range(self.swarm_size):
            # for j in range(self.n_params):
            #     if self.steps[j] is None:  # 如果参数是离散型，则随机选择其中一个可能取值
            #         self.position[i, j] = np.random.choice(self.ranges[j])
            #     else:  # 如果参数是连续型，则在取值范围内随机生成一个可能取值，并按照步长进行四舍五入
            #         self.position[i, j] = round(
            #             np.random.uniform(self.ranges[j][0], self.ranges[j][1]) / self.steps[j]) * self.steps[j]
            #         # self.position[i, j] = round(
            #         #     np.random.uniform(self.ranges[j][0], self.ranges[j][1]) / self.steps[j]) * self.steps[j]
            self.velocity[i, :] = np.random.uniform(-1, 1, self.n_params)  # 在[-1,1]之间随机生成一个速度向量
            print("self.position[i, :]",self.position[i, :])
            self.fitness[i] = self.objective_function(self.position[i, :],
                                                      self.circle_time,
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
    def objective_function(self, x, circle_time, disease_microbe_matrix, disease_sim, microbe_sim, seed, epoch, adjdp,
                           bestAUC):
        emb_dim, dp= int(x[0]), x[1]
        # 0.0005
        # , x[2], int(x[3])
        lr, CNNLayer = 0.01, 2
        # emb_dim, dp, lr, CNNLayer = 256, 0.0005, 0.01, 2
        print("emb_dim, dp, lr, CNNLayer", emb_dim, dp, lr, CNNLayer)
        result = np.zeros((circle_time, 7), float)
        for i in range(circle_time):
            # print("circle_time",circle_time)
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
        # print("2 pso optimize")

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

def PredictScore(train_disease_microbe_matrix, disease_matrix, microbe_matrix, seed, epochs, emb_dim, dp, lr,  adjdp,cvnum,CNNLayer):
    # print("PredictScore")
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_disease_microbe_matrix, disease_matrix, microbe_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_disease_microbe_matrix.sum()
    X = constructNet(train_disease_microbe_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]#331
    features_nonzero = features[1].shape[0]#720
    adj_orig = train_disease_microbe_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))


    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]


    biases = adj_to_bias(adj, [num_features], nhood=1)

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=()),
           }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_disease_microbe_matrix.shape[0],CNNLayer, name='LAGCN')
    # print("model.summary()",model.summary())
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_disease_microbe_matrix.shape[0], num_v=train_disease_microbe_matrix.shape[1], association_nam=association_nam)
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)


    trained_att_value = sess.run(model.att_weights)

    print("Trained att values:", trained_att_value)





    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})



    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    #add by wu
    disease_len = disease_microbe_matrix.shape[0]
    microbe_len = disease_microbe_matrix.shape[1]
    predict_y_proba = res.reshape(disease_len, microbe_len)  # 39行疾病与292个微生物的关联预测
    np.savetxt('predict_y_proba'+str(cvnum)+'.csv', predict_y_proba, delimiter=',', fmt='%5f')


    saver.save(sess, './Model2/model_'+str(cvnum)+'.ckpt')
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.close()
    return res

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs =1
    mt = np.empty(adj.shape)
    for i in range(mt.shape[0]):
        for j in range(mt.shape[1]):
            if mt[i][j] > 0.0:
                mt[i][j] = 1.0
    return -1e9 * (1.0 - mt)


def cross_validation_experiment(disease_microbe_matrix, disease_matrix, microbe_matrix, seed, epochs, emb_dim, dp, lr, adjdp, bestAUC,CNNLayer):
    index_matrix = np.mat(np.where(disease_microbe_matrix == 1))#具有关联关系的疾病和微生物的索引，index_matrix 为2行450列，第一行为每个具有关联的疾病的索引，对应的微生物索引在第二行中
    association_nam = index_matrix.shape[1]#450

    random_index = index_matrix.T.tolist()#【0，168】 【1， 108】具有关联的疾病微生物对
    random.seed(seed)
    random.shuffle(random_index)#关联对打乱
    k_folds = 5#5
    CV_size = int(association_nam / k_folds)#每折关联数量 90
    temp = np.array(random_index[:association_nam - association_nam %
                                 k_folds]).reshape(k_folds, CV_size,  -1).tolist()#temp为分成5组后的list关联对，-1表示维数自动计算，5*90*2 association_nam %k_folds 余数，即未能完全等分的，
    temp[k_folds - 1] = temp[k_folds - 1] + \
        random_index[association_nam - association_nam % k_folds:]#把多余的放到最后一个fold组中
    random_index = temp#random_index 5个list，每个里面是关联对，列的形式，每行一个关联对
    metric = np.zeros((1, 7))
    print("seed=%d, evaluating microbe-disease...." % (seed))
    aucetc = np.zeros((k_folds, 1), dtype=float)
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(disease_microbe_matrix, copy=True)#39*292
        train_matrix[tuple(np.array(random_index[k]).T)] = 0#将第k组关联置零，
        disease_len = disease_microbe_matrix.shape[0]#39
        microbe_len = disease_microbe_matrix.shape[1]#292
        disease_disease_res = PredictScore(
            train_matrix, disease_matrix, microbe_matrix, seed, epochs, emb_dim, dp, lr,  adjdp,k,CNNLayer)#train_matrix 第kfold关联置零后，drugmatrix：39*39，microbe_matrix：292*292，
        predict_y_proba = disease_disease_res.reshape(disease_len, microbe_len)#39行疾病与292个微生物的关联预测
        metric_tmp = cv_model_evaluate(
            disease_microbe_matrix, predict_y_proba, train_matrix)
        # print(metric_tmp)
        AUC=metric_tmp

        if AUC > bestAUC:
            bestAUC = AUC

        aucetc[k,:]=metric_tmp
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print('__________Average performance of cross validation__________')
    print(np.mean(aucetc,axis=0))
    print('Performance for each cv')
    print(aucetc)

    return np.mean(aucetc,axis=0)


if __name__ == "__main__":


    disease_sim_com = np.loadtxt('../data/HMDAD/dis_sim.csv', delimiter=',')#0923
    microbe_sim_com = np.loadtxt('../data/HMDAD/microbe_sim.csv', delimiter=',')
    disease_microbe_matrix = np.loadtxt('../data/HMDAD/dis_microbe.csv', delimiter=',')




    disease_sim_cosine = np.loadtxt('../data/HMDAD/disease_cosine_wuchy.csv', delimiter=',')
    microbe_sim_cosine = np.loadtxt('../data/HMDAD/microbes_cosine_wuchy.csv', delimiter=',')


    disease_sim_gaussian_symptom = np.loadtxt('../data/HMDAD/disease_gaussian_symptom_wuchy.csv', delimiter=',')
    microbe_sim_gaussian = np.loadtxt('../data/HMDAD/microbes_gaussian_wuchy.csv', delimiter=',')
    # disease_microbe_matrix = np.loadtxt('../data/HMDAD/dis_microbe.csv', delimiter=',')

    disease_sim_jaccard = np.loadtxt('../data/HMDAD/disease_jaccard_wuchy.csv', delimiter=',')
    microbe_sim_jaccard = np.loadtxt('../data/HMDAD/microbes_jaccard_wuchy.csv', delimiter=',')

    disease_sim = np.maximum(np.maximum(disease_sim_cosine,disease_sim_jaccard),np.maximum(disease_sim_gaussian_symptom,disease_sim_com))
    microbe_sim = np.maximum(np.maximum(microbe_sim_cosine,microbe_sim_jaccard),np.maximum(microbe_sim_gaussian,microbe_sim_com))
    epoch = 5200
    emb_dim = 256
    lr = 0.01
    adjdp = 0.3
    dp = 0.0005
    simw = 6
    simw2 = 6

    average_result = np.zeros((1, 7), float)
    std_result = np.zeros((1, 7), float)
    circle_time = 10
    result = np.zeros((circle_time, 7), float)
    bestAUC = 0
    seed = 0
    for i in range(circle_time):
        seed = i
        # seed = 0
        result[i,:]= cross_validation_experiment(
            disease_microbe_matrix, disease_sim*simw, microbe_sim*simw2, seed, epoch, emb_dim, dp, lr, adjdp, bestAUC,3)
    print(result)
    # average_result[:,:] = np.average(result[:,:]) #result / circle_time
    # 计算每列的均值
    column_means = result.mean(axis=0)
    # 打印每列的均值
    print("每列的均值为：", column_means)
    # 计算每列的标准差
    column_stddevs = result.std(axis=0)

    # 打印每列的标准差
    print("每列的标准差为：", column_stddevs)

    #=============================================================

    # gbest_position, modelname = ParticleSwarmOptimization(circle_time=circle_time,
    #     disease_microbe_matrix=disease_microbe_matrix,
    #     disease_sim=disease_sim * simw,
    #     microbe_sim=microbe_sim * simw2,
    #     seed=0,
    #     epch= epoch,
    #     adjdp=adjdp,
    #     bestAUC=bestAUC,swarm_size=20, max_iter=5,  w=0.9, c1=2.05, c2=2.05).optimize()


    #=======================================================================
    # # 设定参数的上下界
    # lb = [10, 0.0, 0.0001,1]  # emb_dim至少为10，dp和lr,CNNlayer的下界
    # ub = [260, 0.1, 0.1,100000]  # emb_dim的上界为100，dp和lr,CNNlayer的上界
    #
    #
    # # 假设这些参数已经定义
    # # example values for other parameters
    # epoch = 100
    # adjdp = 0.5
    # bestAUC = 0.85
    #
    # # 创建绑定了额外参数的目标函数
    # bound_objective_function = partial(
    #     objective_function,
    #     circle_time=circle_time,
    #     disease_microbe_matrix=disease_microbe_matrix,
    #     disease_sim=disease_sim * simw,
    #     microbe_sim=microbe_sim * simw2,
    #     seed=0,
    #     epoch=epoch,
    #     adjdp=adjdp,
    #     bestAUC=bestAUC
    # )
    # # 调整迭代次数、粒子群数量、惯性权重
    # options = {'maxiter': 1000, 'swarmsize': 50, 'omega': 0.5}
    #
    # # 调用 PSO 算法
    # xopt, fopt = pso(bound_objective_function, lb, ub, swarmsize=300, maxiter=5000,omega=0.5)
    #
    #
    # print("最优参数：", xopt)
    # print("最优目标函数值：", fopt)
    #


