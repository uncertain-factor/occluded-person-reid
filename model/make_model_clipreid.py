import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# CLIP的文本编码器
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer  # transformer编码器
        self.positional_embedding = clip_model.positional_embedding  # 位置嵌入
        self.ln_final = clip_model.ln_final  # 层归一化层
        self.text_projection = clip_model.text_projection  # 投影层
        self.dtype = clip_model.dtype  # 数据类型

    # 输入一批prompts和相应的tokenized_prompts，经过transformer的encoder编码得到一批[EOS]，视为该批文本的特征
    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x的形状为[batch_size, n_ctx, transformer.width]
        # 从x的每一行取出最后一个维度的值[EOS]，与投影矩阵相乘得到x作为输入的x的文本特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME  # NAME: 'ViT-B-16'
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        # 投影后的纬度为512
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        # 创建全连接层（纬度768）和对应的投影层（纬度512），初始化权重参数
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        # 创建归一化层（纬度768）和对应的投影层（纬度512），初始化权重参数
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        # 窗口patch数量 = (图像高度 - 窗口高度) / 步幅大小 + 1
        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0] - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1] - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")
        # 将clip预训练好的vit模型传给图像编码器，作为模型image部分
        self.image_encoder = clip_model.visual
        # no
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(camera_num))
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=.02)
            print('camera number is : {}'.format(view_num))

        dataset_name = cfg.DATASETS.NAMES
        # 创建模型的文本提示优化模型和文本编码器，作为模型text部分
        self.prompt_learner = PromptLearner(num_classes, dataset_name, clip_model.dtype, clip_model.token_embedding)
        self.text_encoder = TextEncoder(clip_model)

    # 可根据不同的输入获取图像特征和文本特征
    def forward(self, x=None, label=None, get_image=False, get_text=False, cam_label=None, view_label=None):
        # 输入一批labels（行人id），通过文本提示优化器prompt learner获取对应的文本prompts模型参数，
        # 将prompts和tokenized_prompts通过文本编码器text encoder获取文本特征并返回
        if get_text == True:
            prompts = self.prompt_learner(label)    # prompts形状为 (labels_size,len_prefix+n_ctx+len_suffix,512)
            # print("point1")
            # print(self.prompt_learner.tokenized_prompts)
            # print("point2")
            text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts)
            return text_features
        # 输入一批图像x，然后获取这批图像经过vit编码后的图像特征投影后的[cls]，作为图像的特征表示并返回
        if get_image == True:
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            if self.model_name == 'RN50':
                return image_features_proj[0]
            elif self.model_name == 'ViT-B-16':
                return image_features_proj[:, 0]

        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x)
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(
                x.shape[0], -1)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]

        elif self.model_name == 'ViT-B-16':
            if cam_label != None and view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
            elif cam_label != None:
                cv_embed = self.sie_coe * self.cv_embed[cam_label]
            elif view_label != None:
                cv_embed = self.sie_coe * self.cv_embed[view_label]
            else:
                cv_embed = None  # yes
            # image_features_last 表示最后一个 Transformer 层的输出特征，
            # image_features 表示所有 Transformer 层的输出特征的集合，
            # 而 image_features_proj 表示经过位置嵌入和类别嵌入处理后的图像表示。
            # 取第一个位置的向量CLS作为特征向量
            image_features_last, image_features, image_features_proj = self.image_encoder(x, None)
            img_feature_last = image_features_last[:, 0]
            img_feature = image_features[:, 0]
            img_feature_proj = image_features_proj[:, 0]
        # 经过归一化层和对应的投影层，获取维度为768的feat和维度为512的feat_proj
        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        # 训练模式，获取图像所有transformer层输出特征以及图像投影特征经过分类器后的预测分数logits
        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            # 返回两个logits连接后的预测分数，图像组合特征（最后一个transformer，所有transformer，经过投影）和图像经过投影的特征
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj
        # 评估模式
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:   # yes
                # 将图像经过所有transformer层后（第一个向量）的图像特征
                # 与图像经过投影层后（第一个向量）的图像特征连接起来作为新的图像特征并返回
                return torch.cat([img_feature, img_feature_proj], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip


def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model


# 文本提示优化器
class PromptLearner(nn.Module):
    def __init__(self, num_class, dataset_name, dtype, token_embedding):
        super().__init__()
        # 确定文本描述的格式
        if dataset_name == "VehicleID" or dataset_name == "veri":
            ctx_init = "A photo of a X X X X vehicle."
        else:
            ctx_init = "A photo of a X X X X person."
        # 输入纬度为512，句子模版为ctx_init，上下文token数量为4
        ctx_dim = 512
        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        # 通用的context数
        n_ctx = 4
        # 将[SOS]+文本+[EOS]转化为数字形式的token列表,,并固定tokens长度为77
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        # 不计算梯度（冻结该部分参数），对文本进行词嵌入，每个词被映射为纬度512的向量
        with torch.no_grad():
            # 句子模板的word embedding，长度为77*512
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # 特定于类的context数（learnable token）
        n_cls_ctx = 4
        # 用正态分布随机初始化张量容器存放每个行人对应的4个512维度的learnable token，形状为（num_class, n_cls_ctx=4, ctx_dim=512），然后
        cls_vectors = torch.empty(num_class, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        self.cls_ctx = nn.Parameter(cls_vectors)  # 将每个行人的learnable tokens添加为模型参数，随着训练被优化

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # 注册模型的固定参数token_prefix和token_suffix，这些参数不会被优化， context = token_prefix + token_suffix 即learnable context的前后缀
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + n_cls_ctx:, :])
        self.num_class = num_class
        self.n_cls_ctx = n_cls_ctx

    # 输入一个labels列表，构造对应的 prompts 模型参数并返回，prompts形状为 (labels_size,len_prefix+n_ctx+len_suffix,512)
    def forward(self, label):
        # 根据label(行人id)获取对应的learnable tokens 的模型参数,label 可能是一批而不是一个
        cls_ctx = self.cls_ctx[label-1]  # （num_class, n_cls_ctx=4, ctx_dim=512）
        b = label.shape[0]  # 获取该批次的labels的数量
        # 扩展了模型中注册的名为 "token_prefix"和 "token_suffix" 的缓冲区，使其与输入批次中的标签数量相匹配。
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        # 拼接前缀，learnable tokens embedding，后缀作为prompts
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                cls_ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        # (labels_size,len_prefix+n_ctx+len_suffix,512)
        return prompts
