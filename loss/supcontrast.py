"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    # 传入一批文本特征和图像特征以及它们的标签，计算对比损失函数
    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0]
        # 根据文本特征和图像特征的类别标签，创建掩码矩阵，正样本对的矩阵位置为1，其余为0
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 
        # 计算文本特征和图像特征的内积并除以温度系数，得到相似程度分数矩阵logits
        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # 对logits矩阵的每一行减去最大的logit分数，保持数值稳定，所有元素符号都取反，最大的logits分数为0
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        # 对logits矩阵取指数得到exp_logits矩阵，所有元素的数值位于0-1之间
        exp_logits = torch.exp(logits)
        # 对原始的相似程度分数矩阵logits矩阵减去取对数后的exp_logits矩阵
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # 只计算匹配的正样本对，然后求和再取均值
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        # 对均值取反得到对比损失loss
        loss = - mean_log_prob_pos.mean()

        return loss