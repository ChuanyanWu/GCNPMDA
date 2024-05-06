import numpy as np
import csv
from datetime import datetime
import calendar

import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.metrics import average_precision_score,precision_recall_curve, auc

def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])
    return auc[0, 0]


def get_tpr(y, y_hat):
    true_positive = sum(yi and yi_hat for yi, yi_hat in zip(y, y_hat))
    actual_positive = sum(y)
    return true_positive / actual_positive
def get_tnr(y, y_hat):
    true_negative = sum(1 - (yi or yi_hat) for yi, yi_hat in zip(y, y_hat))
    actual_negative = len(y) - sum(y)
    return true_negative / actual_negative

def get_roc(y, y_hat_prob):
    thresholds = sorted(set(y_hat_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_hat_prob >= threshold) for yi_hat_prob in y_hat_prob]
        ret.append([get_tpr(y, y_hat), 1 - get_tnr(y, y_hat)])
    return ret

def get_auc(y, y_hat_prob):
    roc = iter(get_roc(y, y_hat_prob))
    tpr_pre, fpr_pre = next(roc)
    auc = 0
    for tpr, fpr in roc:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        tpr_pre = tpr
        fpr_pre = fpr
    return auc

def cv_model_evaluate(interaction_matrix, predict_matrix, train_matrix):
    test_index = np.where(train_matrix == 0)
    real_score = interaction_matrix[test_index]
    predict_score = predict_matrix[test_index]
    fpr, tpr, thresholds = roc_curve(real_score, predict_score)

    now = datetime.now()

    time_string = now.strftime("%Y-%m-%d %H%M%S")
    filename = "logs/fprtpr_ " + time_string + ".csv"

    plt.figure()

    plt.plot(fpr, tpr, label='ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('./img/pic-{}.png'.format(time_string ))

    # plt.show()
    return get_metrics(real_score, predict_score)
