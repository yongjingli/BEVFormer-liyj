from mmdet3d.models import build_model ,build_backbone, build_neck, build_head

import torch.nn as nn
import torch
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
from mmdet3d.apis import inference_multi_modality_detector, init_model
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)

from mmcv.image import tensor2imgs
from types import MethodType
from thop import profile
from thop import clever_format
from mmcv.cnn.utils.flops_counter import add_flops_counting_methods

from mmdet3d.models import builder
from mmdet.models.utils import build_transformer
# /home/dell/liyongjing/programs/BEVFormer-liyj/mmdetection3d/mmdet3d/models/builder.py
# from .. import builder
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
import copy
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', default=None, help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def init(config, checkpoint):
    args = parse_args()
    args.config = config
    args.checkpoint = checkpoint

    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

    # if args.eval and args.format_only:
    #     raise ValueError('--eval and --format_only cannot be both specified')
    #
    # if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    #     raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    return cfg


class ModelBackbone(nn.Module):
    def __init__(self, cfg):
        super(ModelBackbone, self).__init__()
        # if img_backbone:
        # img_backbone = {'type': 'ResNet', 'depth': 101,
        #                  'num_stages': 4, 'out_indices': (1, 2, 3),
        #                  'frozen_stages': 1, 'norm_cfg': {'type': 'BN2d', 'requires_grad': False},
        #                  'norm_eval': True, 'style': 'caffe',
        #                  'dcn': {'type': 'DCNv2', 'deform_groups': 1, 'fallback_on_stride': False},
        #                  'stage_with_dcn': (False, False, True, True)}
        img_backbone_cfg = cfg['model']['img_backbone']
        self.img_backbone = build_backbone(img_backbone_cfg)
        # if img_neck is not None:

        # img_neck = {'type': 'FPN', 'in_channels': [512, 1024, 2048], 'out_channels': 256, 'start_level': 0, 'add_extra_convs': 'on_output', 'num_outs': 4, 'relu_before_extra_convs': True}
        img_neck_cfg = cfg['model']['img_neck']
        self.img_neck = build_neck(img_neck_cfg)

    def forward(self, img):
        img_feats = self.img_backbone(img)
        img_feats = self.img_neck(img_feats)
        return img_feats

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


class ModelBEVEncoder(nn.Module):
    def __init__(self, cfg):
        super(ModelBEVEncoder, self).__init__()
        encoder = cfg['model']['pts_bbox_head']['transformer']['encoder']
        self.encoder = build_transformer_layer_sequence(encoder)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img


class ModelBEVDecodercoder(nn.Module):
    def __init__(self, cfg):
        super(ModelBEVDecodercoder, self).__init__()

        decoder = cfg['model']['pts_bbox_head']['transformer']['decoder']
        self.decoder = build_transformer_layer_sequence(decoder)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img

class ModelBEVTransformer(nn.Module):
    def __init__(self, cfg):
        super(ModelBEVTransformer, self).__init__()

        transformer = cfg['model']['pts_bbox_head']['transformer']
        self.transformer = build_transformer(transformer)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img


class ModelBEVHead(nn.Module):
    def __init__(self, cfg):
        super(ModelBEVHead, self).__init__()
        # pts_bbox_head = {'type': 'BEVFormerHead', 'bev_h': 200, 'bev_w': 200, 'num_query': 900, 'num_classes': 10,
        #                  'in_channels': 256, 'sync_cls_avg_factor': True, 'with_box_refine': True,
        #                  'as_two_stage': False, 'transformer': {'type': 'PerceptionTransformer', 'rotate_prev_bev': True, 'use_shift': True, 'use_can_bus': True, 'embed_dims': 256,
        #                                                         'encoder': {'type': 'BEVFormerEncoder', 'num_layers': 6, 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'num_points_in_pillar': 4, 'return_intermediate': False, 'transformerlayers': {'type': 'BEVFormerLayer', 'attn_cfgs': [{'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1}, {'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}, 'decoder': {'type': 'DetectionTransformerDecoder', 'num_layers': 6, 'return_intermediate': True, 'transformerlayers': {'type': 'DetrTransformerDecoderLayer', 'attn_cfgs': [{'type': 'MultiheadAttention', 'embed_dims': 256, 'num_heads': 8, 'dropout': 0.1}, {'type': 'CustomMSDeformableAttention', 'embed_dims': 256, 'num_levels': 1}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')}}}, 'bbox_coder': {'type': 'NMSFreeCoder', 'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'max_num': 300, 'voxel_size': [0.2, 0.2, 8], 'num_classes': 10}, 'positional_encoding': {'type': 'LearnedPositionalEncoding', 'num_feats': 128, 'row_num_embed': 200, 'col_num_embed': 200}, 'loss_cls': {'type': 'FocalLoss', 'use_sigmoid': True, 'gamma': 2.0, 'alpha': 0.25, 'loss_weight': 2.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 0.25}, 'loss_iou': {'type': 'GIoULoss', 'loss_weight': 0.0}, 'train_cfg': None, 'test_cfg': None}
        pts_bbox_head = cfg['model']['pts_bbox_head']
        pts_bbox_head['train_cfg'] = None
        pts_bbox_head['test_cfg'] = None

        self.pts_bbox_head = builder.build_head(pts_bbox_head)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img

class ModelPositionalEncoding(nn.Module):
    def __init__(self, cfg):
        super(ModelPositionalEncoding, self).__init__()
        positional_encoding = cfg['model']['pts_bbox_head']['positional_encoding']
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img

class ModelBBoxcoder(nn.Module):
    def __init__(self, cfg):
        super(ModelBBoxcoder, self).__init__()
        bbox_coder = cfg['model']['pts_bbox_head']['bbox_coder']
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img

class ModelRegCls(nn.Module):
    def __init__(self, cfg):
        super(ModelRegCls, self).__init__()
        self.num_reg_fcs = 2
        self.embed_dims = 256
        self.cls_out_channels = 10
        self.num_reg_fcs = 2
        self.code_size = 10
        self.with_box_refine = True
        self.as_two_stage = False
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        # num_pred = (self.transformer.decoder.num_layers + 1) if \
        #     self.as_two_stage else self.transformer.decoder.num_layers
        num_pred = 6

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        # if not self.as_two_stage:
        #     self.bev_embedding = nn.Embedding(
        #         self.bev_h * self.bev_w, self.embed_dims)
        #     self.query_embedding = nn.Embedding(self.num_query,
        #                                         self.embed_dims * 2)

        # bbox_coder = cfg['model']['pts_bbox_head']['bbox_coder']
        # self.bbox_coder = build_bbox_coder(bbox_coder)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img


class ModelNorms(nn.Module):
    def __init__(self, cfg):
        super(ModelNorms, self).__init__()
        norm_cfg = {'type': 'LN'}
        num_norms = 3
        self.embed_dims = 256
        self.norms = ModuleList()
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])
        # bbox_coder = cfg['model']['pts_bbox_head']['bbox_coder']
        # self.bbox_coder = build_bbox_coder(bbox_coder)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img


class ModelFFN(nn.Module):
    def __init__(self, cfg):
        super(ModelFFN, self).__init__()
        self.ffns = ModuleList()
        self.embed_dims = 256
        num_ffns = 1
        ffn_cfgs = [{'type': 'FFN', 'embed_dims': 256, 'feedforward_channels': 512, 'num_fcs': 2, 'ffn_drop': 0.1, 'act_cfg': {'type': 'ReLU', 'inplace': True}}]

        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims

            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index]))

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img


class ModelMultiheadAttention(nn.Module):
    def __init__(self, cfg):
        super(ModelMultiheadAttention, self).__init__()

        embed_dims = 256
        num_heads = 8
        dropout = 0.1
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout)

    def forward(self, img):
        # img_feats = self.img_backbone(img)
        # img_feats = self.img_neck(img_feats)
        return img


def debug_model(cfg, in_tensort):
    # backbone
    model = ModelBackbone(cfg)

    # bev head = 10.24 + 0.46 + 0.8 +0.8 + 0.035 + 0.0015 + 0.001 + 4.94 + 4.1 = 21.378M
    # model = ModelBEVHead(cfg)

    # bev transformer = bev encode + bev decode = 9.083M
    # model = ModelBEVTransformer(cfg)

    # bev encode = 4.941M
    # model = ModelBEVEncoder(cfg)

    # bev decode = 4.103M
    # model = ModelBEVDecodercoder(cfg)

    # position encode = 0 M
    # model = ModelPositionalEncoding(cfg)

    # bbox coder = 0 M
    # model = ModelBBoxcoder(cfg)

    # cls reg = 1.616M
    # model = ModelRegCls(cfg)

    # norms = 1.536K
    # model = ModelNorms(cfg)

    # ffn = 262.912K
    # model = ModelFFN(cfg)

    # multi-head-attention
    # model = ModelMultiheadAttention(cfg)

    print("model:", model)

    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()

    _ = flops_model(in_tensort)
    # flops_count, params_count = flops_model.compute_average_flops_cost()
    flops_count, params_count = flops_model.compute_average_flops_cost()

    img_num = in_tensort.shape[0]
    flops_count = flops_count * img_num
    # flops_count, params_count = flops_model.get_model_complexity_info()
    flops_model.stop_flops_count()

    flops, params = flops_count, params_count

    flops, params = clever_format([flops, params], "%.3f")
    print("img:", in_tensort.shape)
    print("flops:", flops)
    print("params:", params)

    # out = model(in_tensort)
    # print(out)


if __name__ == "__main__":
    print("Start")
    # bevformer tiny
    cfg_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_tiny.py"
    checkpoint_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_tiny_epoch_24.pth"
    in_tensort = torch.randn([6, 3, 480, 800])

    # bevformer small
    # cfg_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_small.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_small_epoch_24.pth"
    # in_tensort = torch.randn([6, 3, 736, 1280])

    # bevformer base
    # cfg_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_base.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_r101_dcn_24ep.pth"
    # in_tensort = torch.randn([6, 3, 928, 1600])
    # in_tensort = torch.randn([1, 3, 224, 224])     # 正常的分类模型输入大小

    cfg = init(cfg_path, checkpoint_path)
    debug_model(cfg, in_tensort)
    print("End")