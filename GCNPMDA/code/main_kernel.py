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



def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp,cvnum):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)#[【drug_matrix drug_dis_matrix】,【drug_dis_matrix.T dis_matrix】]331*331
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()#360条关联，在此训练集中，第kfold关联被置为0了，所以不是全部的450个1
    X = constructNet(train_drug_dis_matrix)#[【drug_matrix drug_dis_matrix】,【drug_dis_matrix.T dis_matrix】]331*331其中drug_matrix和dis_matrix为0矩阵
    features = sparse_to_tuple(sp.csr_matrix(X))#sparse_to_tuple返回一个稀疏矩阵的非0值坐标  features[0] 关联对  features[1] label1，features[2] 331*331
    num_features = features[2][1]#331
    features_nonzero = features[1].shape[0]#720
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))#coords, values, shape（有值位置的坐标，值，特征向量的shape）三要素作为元组。
    # adj_orj[0] 是关联对的node[  0, 168],       [  2, 168],       [  3,  60],。。。adj_orj[1]是1111111  adj_orj[2]是大小 39*292

    adj_norm = preprocess_graph(adj)## 计算标准化的邻接矩阵：根号D * A * 根号D
    adj_nonzero = adj_norm[1].shape[0]

    #add by wuchy
    biases = adj_to_bias(adj, [num_features], nhood=1)

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=()),
           }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='LAGCN',)
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),#将张量展平为1维，先行后列
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1], association_nam=association_nam)
    saver = tf.train.Saver()  # 声明tf.train.Saver类用于保存模型
    sess = tf.Session()#开启一个会话模式，因为后续我们需要通过sess.run()来执行计算图。而global_variables_initializer()则是用于对之前所有定义的Variable()进行初始化赋值操作（声明的时候并没有完成赋值操作）。
    sess.run(tf.global_variables_initializer())#全局变量初始化 # 所有节点初始化，一句代码即可
    for epoch in range(epochs):
        feed_dict = dict()#feed_dict 对于前面定义的所有的placeholder，在启动计算图时都需要喂入相应的真实数据。在Tensorflow中，我们将以一个字典的形式把所有占位符需要的东西传进去。注意，字典的key就是占位符的名称，value就是需要传入的值。
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        # feed_dict.update({placeholders['bias_in']: biases})#add by wuchy
        # feed_dict.update({placeholders['ffd_drop']: 0.1})  # add by wuchy
        # feed_dict.update({placeholders['attn_drop']: 0.1})  # add by wuchy
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)#由于我们需要更新网络权重，所以要执行train_op这个优化器操作。同时，我们需要输出查看具体的损失值，所以要将执行opt_op操作，计算optimizer.minimize(
           # cost的计算；最后，我们用avg_cost来接收返回的损失值， _来忽略traip_op返回的值。
        # if epoch % 100 == 0:# 每隔100轮迭代输出一次信息
        #     feed_dict.update({placeholders['dropout']: 0})
        #     feed_dict.update({placeholders['adjdp']: 0})
        #     # res = sess.run(model.reconstructions, feed_dict=feed_dict)
        #     # print("Epoch:", '%04d' % (epoch + 1),
        #     #       "train_loss=", "{:.5f}".format(avg_cost))



    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})



    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    #add by wu
    drug_len = drug_dis_matrix.shape[0]
    dis_len = drug_dis_matrix.shape[1]
    predict_y_proba = res.reshape(drug_len, dis_len)  # 39行疾病与292个微生物的关联预测
    np.savetxt('predict_y_proba'+str(cvnum)+'.csv', predict_y_proba, delimiter=',', fmt='%5f')

    # model.save_model('./Model2/GCNModel_CV_'+str(cvnum)+'.ckpt')
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


def cross_validation_experiment(drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr, adjdp, bestAUC):
    index_matrix = np.mat(np.where(drug_dis_matrix == 1))#具有关联关系的疾病和微生物的索引，index_matrix 为2行450列，第一行为每个具有关联的疾病的索引，对应的微生物索引在第二行中
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
    aupraucetc = np.zeros((k_folds, 7), dtype=float)
    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k+1))
        train_matrix = np.matrix(drug_dis_matrix, copy=True)#39*292
        train_matrix[tuple(np.array(random_index[k]).T)] = 0#将第k组关联置零，
        drug_len = drug_dis_matrix.shape[0]#39
        dis_len = drug_dis_matrix.shape[1]#292
        drug_disease_res = PredictScore(
            train_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp,k)#train_matrix 第kfold关联置零后，drugmatrix：39*39，dis_matrix：292*292，
        predict_y_proba = drug_disease_res.reshape(drug_len, dis_len)#39行疾病与292个微生物的关联预测
        metric_tmp = cv_model_evaluate(
            drug_dis_matrix, predict_y_proba, train_matrix)
        print(metric_tmp)
        AUC=metric_tmp[1]

        if AUC > bestAUC:
            bestAUC = AUC

            # os.rename('./Model2/my_model.pkl', './Model2/my_model_best.pkl')
            # os.rename('./ Model2 / model.ckpt', './ Model2 / model_best.ckpt')
            # os.rename('./Model2/my_model_weights.h5', './Model2/my_model_weights_best.h5l')
        aupraucetc[k,:]=metric_tmp
        metric += metric_tmp
        del train_matrix
        gc.collect()
    print('Average performance of cross validation')
    print(metric / k_folds)
    print('Performance for each cv')
    print(aupraucetc)
    # 求⽅差
    var = np.var(aupraucetc, axis=0)
    print('方差')
    print(var)
    # 求标准差
    std = np.std(aupraucetc, axis=0)
    print('标准差')
    print(std)
    metric = np.array(metric / k_folds)
    print(metric_tmp)
    #del train_matrix
    return metric_tmp


if __name__ == "__main__":
    print(device_lib.list_local_devices())
    # drug_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/HMDAD/dis_sim.csv', delimiter=',')#0923
    # dis_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/HMDAD/microbe_sim.csv', delimiter=',')
    # drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/HMDAD/dis_microbe.csv', delimiter=',')
    # epoch = 4000#4000 #5200
    # emb_dim = 256#256最好
    # lr = 0.01#0.01
    # adjdp = 0.3#0.3#0.6
    # dp = 0.0005#0.0005#0.0001#0.0001#0.4
    # simw = 6#6#4.5时aupr最好的时候为0.91#5.5 0.86平均aupr
    # simw2 = 6  # 6#4.5时aupr最好的时候为0.91#5.5 0.86平均aupr
    # result = np.zeros((1, 7), float)
    # average_result = np.zeros((1, 7), float)
    # circle_time = 1
    # bestAUC = 0
    # DataSet ="HMDAD"
    DataSet = "Disbiome"





    drug_sim_com = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/dis_sim.csv', delimiter=',')#0923
    dis_sim_com = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbe_sim.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/dis_microbe.csv', delimiter=',')

    drug_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/disease_cosine_wuchy.csv', delimiter=',')
    dis_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbes_cosine_wuchy.csv', delimiter=',')
    drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/dis_microbe.csv', delimiter=',')

    # drug_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/disease_cosine_wuchy.csv', delimiter=',')
    # dis_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbes_cosine_wuchy.csv', delimiter=',')
    # drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/dis_microbe.csv', delimiter=',')
    drug_sim_cosine = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/disease_cosine_wuchy.csv', delimiter=',')
    dis_sim_cosine = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbes_cosine_wuchy.csv', delimiter=',')
    # drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/dis_microbe.csv', delimiter=',')

    # drug_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/disease_jaccard_wuchy.csv', delimiter=',')
    # dis_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbes_jaccard_wuchy.csv', delimiter=',')
    # drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/dis_microbe.csv', delimiter=',')

    drug_sim_gaussian_symptom = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/disease_gaussian_symptom_wuchy.csv', delimiter=',')
    dis_sim_gaussian = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbes_gaussian_wuchy.csv', delimiter=',')
    # drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/dis_microbe.csv', delimiter=',')

    # drug_sim_cosine = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/disease_cosine_wuchy.csv', delimiter=',')
    # dis_sim_cosine = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbes_cosine_wuchy.csv', delimiter=',')

    drug_sim_jaccard = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/disease_jaccard_wuchy.csv', delimiter=',')
    dis_sim_jaccard = np.loadtxt('C:/Pythonworkspace/LAGCN/data/'+DataSet+'/microbes_jaccard_wuchy.csv', delimiter=',')

    # drug_sim = (drug_sim_cosine+drug_sim_gaussian_symptom)/2
    # dis_sim = (dis_sim_cosine+dis_sim_gaussian)/2
    drug_sim = np.maximum(drug_sim_cosine,drug_sim_jaccard,drug_sim_gaussian_symptom )
    dis_sim =  np.maximum(dis_sim_cosine,dis_sim_jaccard,dis_sim_gaussian)



    # drug_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/Disbiome/dis_sim.csv', delimiter=',')
    # dis_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/Disbiome/microbe_sim.csv', delimiter=',')
    # drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/Disbiome/dis_microbe.csv', delimiter=',')

    # drug_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/Disbiome/dis_sim.csv', delimiter=',')
    # dis_sim = np.loadtxt('C:/Pythonworkspace/LAGCN/data/Disbiome/microbe_sim.csv', delimiter=',')
    # drug_dis_matrix = np.loadtxt('C:/Pythonworkspace/LAGCN/data/Disbiome/dis_microbe.csv', delimiter=',')
    epoch = 5200#4000 #5200
    emb_dim = 256#256最好
    lr = 0.01#0.01
    adjdp = 0.3#0.3#0.6
    dp = 0.0005#0.0005#0.0001#0.0001#0.4
    simw = 6#6#4.5时aupr最好的时候为0.91#5.5 0.86平均aupr
    simw2 = 6  # 6#4.5时aupr最好的时候为0.91#5.5 0.86平均aupr
    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 1
    bestAUC = 0
    for i in range(circle_time):
        result += cross_validation_experiment(
            drug_dis_matrix, drug_sim*simw, dis_sim*simw2, i, epoch, emb_dim, dp, lr, adjdp, bestAUC)
    average_result = result / circle_time
    print(average_result)
