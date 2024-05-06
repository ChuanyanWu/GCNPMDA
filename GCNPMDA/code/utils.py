import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = tf.random_uniform(
        [input_dim, output_dim],
        minval=-init_range,
        maxval=init_range,
        dtype=tf.float32
    )

    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out*(1./keep_prob)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_nomalized = adj_.dot(
        degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_nomalized = adj_nomalized.tocoo()
    return sparse_to_tuple(adj_nomalized)


def constructNet(drug_dis_matrix):
    drug_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[0], drug_dis_matrix.shape[0]), dtype=np.int8))#39*39
    dis_matrix = np.matrix(
        np.zeros((drug_dis_matrix.shape[1], drug_dis_matrix.shape[1]), dtype=np.int8))#292*292

    mat1 = np.hstack((drug_matrix, drug_dis_matrix))#按水平方向(列顺序)堆叠数组构成一个新的数组【drug_matrix drug_dis_matrix】39*331
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))#【drug_dis_matrix.T dis_matrix】292*331
    adj = np.vstack((mat1, mat2))#[【drug_matrix drug_dis_matrix】,【drug_dis_matrix.T dis_matrix】]331*331
    #按垂直方向(行顺序)堆叠数组构成一个新的数组 堆叠的数组需要具有相同的维度
    return adj


def constructHNet(drug_dis_matrix, drug_matrix, dis_matrix):
    mat1 = np.hstack((drug_matrix, drug_dis_matrix))#按水平方向(列顺序)堆叠数组构成一个新的数组【drug_matrix drug_dis_matrix】39*331
    mat2 = np.hstack((drug_dis_matrix.T, dis_matrix))#按水平方向(列顺序)堆叠数组构成一个新的数组【drug_dis_matrix.T dis_matrix】292*331
    return np.vstack((mat1, mat2))#按垂直方向(行顺序)堆叠数组构成一个新的数组 堆叠的数组需要具有相同的维度[【drug_matrix drug_dis_matrix】,【drug_dis_matrix.T dis_matrix】]331*331
