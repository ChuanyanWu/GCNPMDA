import numpy as np

from clr import cyclic_learning_rate
import tensorflow as tf
import tensorflow as tf

import tensorflow as tf


def focal_loss(logits, label, a, r):
    '''
    :param logits: [batch size,num_classes] score value
    :param label: [batch size,num_classes] gt value
    :param a: generally be 0.5
    :param r: generally be 0.9
    :return: scalar loss value of a batch
    '''
    p_1 = - a * np.power(1 - logits, r) * np.math.log2(logits) * label
    p_0 = - (1 - a) * np.power(logits, r) * np.math.log2(1 - logits) * (1 - label)
    return (p_1 + p_0).sum()

class Optimizer():
    def __init__(self, model, preds, labels, lr, num_u, num_v, association_nam):
        norm =num_u*num_v / float((num_u*num_v-association_nam) * 2)#
        preds_sub = preds
        labels_sub = labels
        pos_weight = float(num_u*num_v-association_nam)/(association_nam)

        global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cyclic_learning_rate(global_step=global_step, learning_rate=lr*0.1,
                                                                                  max_lr=lr, mode='exp_range', gamma=0.995)
                                                ,beta1=0.28,beta2=0.5999,epsilon=1e-08,use_locking=False,name='Adam')  # .995))        #beta1=0.29,beta2=0.5999,#  ,beta1=0.08,beta2=0.58,,beta1=0.28,beta2=0.5999,








        self.cost =(norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                logits=preds_sub, targets=labels_sub, pos_weight=pos_weight)))



        self.opt_op = self.optimizer.minimize( self.cost, global_step=global_step,)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)



    import numpy as np






