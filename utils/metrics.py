import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # 对 distmat 的每一行（即每个查询样本到所有参考样本的距离）进行排序，得到排序后的索引 indices。
    indices = np.argsort(distmat, axis=1)
    # 创建一个匹配矩阵 matches。对于每个查询样本，检查其对应的画廊样本 ID 是否与查询样本 ID 相同。
    # 如果相同，则 matches 中对应位置为 1，否则为 0。
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # 存储累计匹配特性（CMC）、平均精度（AP）和有效查询样本的数量。
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    # q_idx为测试集样本的迭代器
    for q_idx in range(num_q):
        # 取出测试样本的行人id和相机id
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # 剔除在参考集中与测试样本的行人id和相机id相同的样本
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # 返回与当前测试样本匹配的有效样本列表，即行人id相同
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        # 计算累计匹配特性（CMC）和平均匹配精度
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
    # 重置计数值
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
    # 获取feats，pids，camids等数据
    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        # feats形状为[batch_num,feat_dim]
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            # 对每一行的feat进行层归一化，对每个维度的元素除以所有元素的平方和的开根
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query 取出测试集的样本特征，pid和camid
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery 取出参考集的样本特征，pid和camid
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:   # yes
            print('=> Computing DistMat with euclidean_distance')
            # 计算查询特征 qf 和参考特征 gf 之间的欧几里得距离，得到距离矩阵 distmat。
            distmat = euclidean_distance(qf, gf)
        # 传入距离矩阵，测试集标签，参考集标签，测试集相机标签，参考集相机标签，计算命中率等指标
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



