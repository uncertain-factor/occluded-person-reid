import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from loss.supcontrast import SupConLoss


# 两个损失函数，一个是loss_func（由三元组损失，交叉熵损失，图文相似度损失,对比损失组成），一个是中心损失center_criterion，内部随机初始化了每个类别的中心点
def make_loss(cfg, num_classes):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048  # 特征向量的维度
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # 创建中心损失函数 loss1
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:  # yes
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # 创建三元组损失函数，margin=0.3 loss2
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # 创建交叉熵损失函数 loss3
        print("label smooth on, numclasses:", num_classes)
    contrast = SupConLoss("cuda")  # 对比损失函数
    # no
    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        # 计算总损失函数的计算，ID_LOSS使用的是交叉熵损失，TRI_LOSS代表三元组损失，I2TLOSS使用的是交叉熵损失，衡量图像特征与文本特征的相似度
        def loss_func(score, feat, target, target_cam, i2tscore=None, whole_img_proj=None, occ_img_proj=None, whole_img_label=None, occ_img_label=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # 如果score是二维列表，代表多个样本（图像到真实标签）的预测分数，计算所有元素的交叉熵，然后计算总和ID_LOSS
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    # 如果score是一个列表，代表单个样本的预测分数，直接计算交叉熵损失，作为ID_LOSS
                    else:
                        ID_LOSS = xent(score, target)
                    # 如果feat是列表，代表每个样本的特征（图像的组合特征），根据样本的标签计算所有样本的三元组损失，然后计算总和TRI_LOSS
                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    # 如果feat是元素，代表一个样本的特征，直接计算三元组损失TRI_LOSS
                    else:
                        TRI_LOSS = triplet(feat, target)[0]
                    # 总损失loss等于交叉熵损失和三元组损失乘以相应的权重参数之和
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    # 传入图像特征与文本特征的相似度（图像到文本类别的预测分数logits），以及真实标签，计算交叉熵损失，再乘以权重加入到loss中
                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss
                    if whole_img_proj !=None and occ_img_proj !=None:
                        # 计算两个方向的损失（从全身到遮挡和从遮挡到全身）
                        loss_i2t = contrast(whole_img_proj, occ_img_proj, whole_img_label, occ_img_label)
                        loss_t2i = contrast(occ_img_proj, whole_img_proj, occ_img_label, whole_img_label)
                        CONTRAST_LOSS = loss_i2t + loss_t2i
                        loss += CONTRAST_LOSS
                    return loss
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
