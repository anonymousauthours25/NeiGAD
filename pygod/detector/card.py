# -*- coding: utf-8 -*-
"""Community-Guided Contrastive Learning with Anomaly-Aware Reconstruction for 
Anomaly Detection on Attributed Networks.(CARD) """
# Author: Yang Wang<yangwang0222@163.com>
# License: BSD 2 clause

import torch
from torch_geometric.nn import GCN

from .base import DeepDetector
from ..nn import CARDBase


class CARD(DeepDetector):
    """
    Community-Guided Contrastive Learning with Anomaly-Aware Reconstruction for 
    Anomaly Detection on Attributed Networks.

    CARD is a contrastive learning based method and utilizes mask reconstruction and community
    information to make anomalies more distinct. This model is train with contrastive loss and 
    local and global attribute reconstruction loss. Random neighbor sampling instead of random walk 
    sampling is used to sample the subgraph corresponding to each node. Since random neighbor sampling 
    cannot accurately control the number of neighbors for each sampling, it may run slower compared to 
    the method implementation in the original paper.

    See:cite:`Wang2024Card` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    subgraph_num_neigh: int, optional
        Number of neighbors in subgraph sampling for each node, Values not exceeding 4 are recommended for efficiency.
        Default: ``4``.
    fp: float, optional
        The balance parameter between the mask autoencoder module and contrastive learning.
        Default: ``0.6``
    gama: float, optional
        The proportion of the local reconstruction in contrastive learning module.
        Default: ``0.5``
    alpha: float, optional
        The proprotion of the community embedding in the conbine_encoder.
        Default: ``0.1``
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs
        Other parameters for the backbone.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N \\times` ``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=2,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 subgraph_num_neigh=4,
                 fp=0.6,
                 gama=0.5,
                 alpha=0.1,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 eval_mode=False,
                 **kwargs):
        super(CARD, self).__init__(hid_dim=hid_dim,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   weight_decay=weight_decay,
                                   act=act,
                                   backbone=backbone,
                                   contamination=contamination,
                                   lr=lr,
                                   epoch=epoch,
                                   gpu=gpu,
                                   batch_size=batch_size,
                                   num_neigh=num_neigh,
                                   verbose=verbose,
                                   save_emb=save_emb,
                                   compile_model=compile_model,
                                   eval_mode=eval_mode,
                                   **kwargs)
        self.subgraph_num_neigh = subgraph_num_neigh
        self.fp = fp
        self.gama = gama
        self.alpha = alpha

    def process_graph(self, data):
        community_adj, self.diff_data = CARDBase.process_graph(data)
        data.community_adj = community_adj.to(self.device)
        self.diff_data = self.diff_data.to(self.device)
        self.diff_data.community_adj = community_adj.to(self.device)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim)

        return CARDBase(in_dim=self.in_dim,
                        subgraph_num_neigh=self.subgraph_num_neigh,
                        fp=self.fp,
                        gama=self.gama,
                        alpha=self.alpha,
                        hid_dim=self.hid_dim,
                        num_layers=self.num_layers,
                        dropout=self.dropout,
                        act=self.act,
                        backbone=self.backbone,
                        **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size

        data.x = data.x.to(self.device)
        data.edge_index = data.edge_index.to(self.device)

        pos_logits, neg_logits, x_, local_x_ = self.model(data)
        diff_pos_logits, diff_neg_logits, _, _ = self.model(
            self.diff_data)

        logits = torch.cat([pos_logits[:batch_size],
                            neg_logits[:batch_size]])
        diff_logits = torch.cat([diff_pos_logits[:batch_size],
                                 diff_neg_logits[:batch_size]])

        con_label = torch.cat([torch.ones(batch_size),
                               torch.zeros(batch_size)]).to(self.device)

        loss, score = self.model.loss_func(
            logits, diff_logits, x_, local_x_, data.x, con_label)

        return loss, score.detach().cpu()