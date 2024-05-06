from typing import List
import tensorflow as tf
from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
from utils import *
from util_gat import layers
import random

class GCNModel():
    #__init__()是构造方法，当我们建立类对象时，首先调用该方法初始化类对象。
    def __init__(self, placeholders, num_features, emb_dim, features_nonzero, adj_nonzero, num_r, CNNlayer,name,  act=tf.nn.leaky_relu):#tf.nn.leaky_relu#act=tf.nn.elu
        self.name = name
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.emb_dim = emb_dim
        self.features_nonzero = features_nonzero
        self.adj_nonzero = adj_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.adjdp = placeholders['adjdp']
        self.act = act
        self.CNNLayer = CNNlayer
        # self.att = tf.Variable([0.25,0.25,0.25,0.25], trainable=True)
        if CNNlayer == 1:
            self.att = tf.Variable(1.0, trainable=True)
        else:
            # self.att = tf.Variable([0.8,0.2], trainable=True)
            # self.att = tf.Variable(tf.random.normal([int(CNNlayer)]), trainable=True)
            self.att = tf.Variable(tf.ones([CNNlayer]) / CNNlayer, trainable=True)
            # self.att =tf.Variable([0.5,0.5], trainable=True)
        self.layers = []
        # print("GCN INIT")
        self.att_weights = self.att

        # Initial sparse convolution layer
        self.layers.append(GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act))

        # Adding more layers
        for i in range(1, self.CNNLayer-1):
            self.layers.append(GraphConvolution(
                name=f'gcn_dense_layer_{i}',
                input_dim=self.emb_dim,
                output_dim=self.emb_dim,
                adj=self.adj,
                dropout=self.dropout,
                act=self.act))

        self.num_r = num_r
        with tf.variable_scope(self.name):
            self.build()


    def build(self):
        self.adj = dropout_sparse(self.adj, 1-self.adjdp, self.adj_nonzero)
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer',
            input_dim=self.input_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            features_nonzero=self.features_nonzero,
            dropout=self.dropout,
            act=self.act)(self.inputs)
        self.hidden2 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,#self.dropout,
            act=self.act)(self.hidden1)
        self.hidden3 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,#self.dropout,
            act=self.act)(self.hidden2)
        self.hidden4 = GraphConvolution(
            name='gcn_dense_layer',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout=self.dropout,#self.dropout,
            act=self.act)(self.hidden3)
        self.hidden5 = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=self.emb_dim,
            output_dim=self.emb_dim,
            adj=self.adj,
            dropout= self.dropout,
            act=self.act)(self.hidden4)


        # # self.embeddings =  self.hidden1*self.att[0]#+self.hidden2 *self.att[1]+self.hidden3 *self.att[2]+ self.emb*self.att[3]#+self.emb*self.att[1]###+self.emb*self.att[2]+self.hidden3 *self.att[3]
        #
        if self.CNNLayer== 1:
            self.att_weights = self.att
            self.embeddings = self.hidden1 * self.att_weights
        else:
            self.att_weights = tf.nn.softmax(self.att)
            self.embeddings =  self.hidden1*self.att_weights[0]+self.hidden2 *self.att_weights[1]+self.hidden3 *self.att[2]#+ self.hidden4*self.att[3]+self.hidden5*self.att[4]#+self.emb*self.att[1]###+self.emb*self.att[2]+self.hidden3 *self.att[3]
        # self.embeddings = self.hidden1 * self.att[0] + self.hidden2 * self.att[1]  # +self.hidden3 *self.att[2]+ self.emb*self.att[3]#+self.emb*self.att[1]###+self.emb*self.att[2]+self.hidden3 *self.att[3]
        #
        # # self.reconstructions = mysvm( input_dim=self.emb_dim, name='gcn_decoder', act=tf.nn.sigmoid)(self.embeddings)
        # print("GCN build")
        # # ======================
        # self.adj = dropout_sparse(self.adj, 1 - self.adjdp, self.adj_nonzero)

        #
        # x = self.layers[0](self.inputs)
        # layer_outputs = [x]
        #
        # for layer in self.layers[1:]:
        #     x = layer(x)
        #     layer_outputs.append(x)
        #
        # if self.CNNLayer == 1:
        #     self.att_weights = tf.constant([1.0], dtype=tf.float32)  # Ensure it's a tensor
        # else:
        #     self.att_weights = tf.nn.softmax(self.att)
        #
        # self.embeddings = 0
        #
        # for i, output in enumerate(layer_outputs):
        #     # print("output",output)
        #     # print("self.att_weights[i]",self.att_weights[i])
        #     self.embeddings += output * self.att_weights[i]
        # ===============================================



        self.reconstructions = InnerProductDecoder(
            name='gcn_decoder',
            input_dim=self.emb_dim, num_r=self.num_r, act=tf.nn.sigmoid)(self.embeddings)#act=tf.nn.sigmoid

    def save_model(self, filepath):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.save(sess, filepath)