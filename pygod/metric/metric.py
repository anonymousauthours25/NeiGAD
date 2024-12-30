# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the outlier detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)
import warnings
import numpy as np

def eval_roc_auc(label, score):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """

    roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc


def eval_recall_at_k(label, score, k=None):
    """
    Recall score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        recall. Default: ``None``.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    recall_at_k = sum(label[score.topk(k).indices]) / sum(label)
    return recall_at_k


def eval_precision_at_k(label, score, k=None):
    """
    Precision score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        precision. Default: ``None``.

    Returns
    -------
    precision_at_k : float
        Precision for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    precision_at_k = sum(label[score.topk(k).indices]) / k
    return precision_at_k


def eval_average_precision(label, score):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    ap = average_precision_score(y_true=label, y_score=score)
    return ap


def eval_f1(label, pred):
    """
    F1 score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier prediction in shape of ``(N, )``.

    Returns
    -------
    f1 : float
        F1 score.
    """

    f1 = f1_score(y_true=label, y_pred=pred)
    return f1


def statistical_parity(pred, sensitive_var_dict):
    """
    Statistical parity of model prediction across different sensitive attribute values

    Parameters
    ----------
    pred : numpy.ndarray
        BINARY Outlier predictions (0 or 1) in shape of ``(N, )``where 1 represents outliers,
        0 represents normal instances.

    sensitive_var_dict: dictionary of key -> numpy.ndarray
        For each value of the sensitive attribute, a list of indexes that correspond to that value. 
        E.g. A list of indices for each gender in the dataset.

    Returns
    -------
    SP : float
        Statistical Parity score (0 to 1), calculated as the maximum rate of prediction y_hat minus minimum
        rate of prediction across all values v of the sensitive attribute X. Lower is better.  
        $\Delta_{SP} = \max_{v\in X}(P(\hat{y}=1 | X=v)) - \min_{v\in X}(P(\hat{y}=1 | X=v))$.
    """
    rates = []
    for v in sensitive_var_dict: # 0.0, 1.0
        rates.append(np.mean(pred[sensitive_var_dict[v]]))
    return max(rates) - min(rates)

def equality_of_odds(pred, true, sensitive_var_dict):
    """
    Equality of odds of model prediction across different sensitive attribute values for nodes that are outliers

    Parameters
    ----------
    pred : numpy.ndarray
        BINARY Outlier predictions (0 or 1) in shape of ``(N, )``where 1 represents outliers,
        0 represents normal instances.

    true : numpy.ndarray
        Labels in shape of ``(N, )``, where 1 represents outliers, 0 represents normal instances.

    sensitive_var_dict: dictionary of key -> numpy.ndarray
        For each value of the sensitive attribute, a list of indexes that correspond to that value. 
        E.g. A list of indices for each gender in the dataset.

    Returns
    -------
    EO : float
        Equality of odds score (0 to 1), calculated as the maximum rate of prediction y_hat minus minimum
        rate of prediction across all values v of the sensitive attribute X that are labelled as outliers.
        Lower is better.  
        $\Delta_{EO} = \max_{v\in X}(P(\hat{y}=1 | X=v, y=1)) - \min_{v\in X}(P(\hat{y}=1 | X=v, y=1))$.
    """
    sens_var = np.zeros_like(pred)
    number_of_values = len(sensitive_var_dict)
    for v in range(number_of_values):
        sens_var[sensitive_var_dict[v]] = v

    full = np.column_stack((pred, true, sens_var))
    only_true = full[full[:, 1] == 1]

    rates = []
    for v in range(number_of_values):
        temp = only_true[only_true[:, 2] == v]
        rates.append(np.mean(temp[:, 0]))

    if len(rates) < 2:
        warnings.warn("Positive labels only exist in <2 sensitive variable categories. EO value meaningless")

    return max(rates) - min(rates)
