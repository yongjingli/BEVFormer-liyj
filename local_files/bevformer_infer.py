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


def model_infer(self, img_metas, img=None, prev_bev=None, rescale=False):
    """Test function without augmentaiton."""
    img_feats = self.extract_feat(img=img, img_metas=img_metas)
    # return img_feats

    bbox_list = [dict() for i in range(len(img_metas))]
    new_prev_bev, bbox_pts = self.simple_test_pts(
        img_feats, img_metas, prev_bev, rescale=rescale)
    for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        result_dict['pts_bbox'] = pts_bbox
    return new_prev_bev, bbox_list


def cal_params_by_mmcv(model, img_metas, img):
    from mmcv.cnn.utils.flops_counter import add_flops_counting_methods
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()

    # input
    # input_shape = (1, 100, 3)
    # try:
    #     batch = torch.ones(()).new_empty(
    #         (1, *input_shape),
    #         dtype=next(flops_model.parameters()).dtype,
    #         device=next(flops_model.parameters()).device)
    # except StopIteration:
    #     # Avoid StopIteration for models which have no parameters,
    #     # like `nn.Relu()`, `nn.AvgPool2d`, etc.
    #     batch = torch.ones(()).new_empty((1, *input_shape))

    _ = flops_model(img_metas, img)
    flops_count, params_count = flops_model.compute_average_flops_cost()
    flops_model.stop_flops_count()

    flops, params = flops_count, params_count
    # print(f'Flops: {flops}\nParams: {params}')
    return flops, params


def bevformer_infer():
    # bevformer tiny
    # cfg_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_tiny.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_tiny_epoch_24.pth"

    # bevformer small
    # cfg_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_small.py"
    # checkpoint_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_small_epoch_24.pth"

    # bevformer base
    cfg_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_base.py"
    checkpoint_path = "/home/dell/liyongjing/programs/BEVFormer-liyj/ckpts/bevformer_r101_dcn_24ep.pth"


    cfg = init(cfg_path, checkpoint_path)

    # init model
    device = 'cuda:0'
    model = init_model(cfg_path, checkpoint_path, device=device)
    # model = MMDataParallel(model, device_ids=[0])

    # init dataloader
    # build the dataloader
    samples_per_gpu = 1
    distributed = False
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # start det
    show = False
    out_dir = "/home/dell/下载/debug"
    show_score_thr = 0.3
    new_prev_bev = None

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        # model = MMDataParallel(model, device_ids=[0])   # 需要配合这个使用，在上方进行设置
        # with torch.no_grad():
        #     result = model(return_loss=False, rescale=True, **data)

        with torch.no_grad():
            img_metas = data["img_metas"][0].data[0]
            img = data['img'][0].data[0]

            img = img.to(device)
            # result = model.simple_test(img_metas, img)
            # new_prev_bev, bbox_list = result

            model.forward = MethodType(model_infer, model)
            if 1:
                result = model(img_metas, img, prev_bev=new_prev_bev)
                new_prev_bev, bbox_list = result
                # exit(1)
            else:
                # 不能先infer，因为会改变img_metas里的数值
                # 计算参数量和计算量
                # flops, params = profile(model, inputs=(img_metas, img))

                # 采用mmcv的方式进行统计，要不然有些操作不计算，如Linear等
                flops, params = cal_params_by_mmcv(model, img_metas, img)
                flops, params = clever_format([flops, params], "%.3f")
                print("img:", img.shape)
                print("flops:", flops)
                print("params:", params)
                exit(1)

        if show and 0:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (Base3DDetector, Base3DSegmentor,
                         SingleStageMono3DDetector)
            if isinstance(model.module, models_3d):
                model.module.show_results(data, result, out_dir=out_dir)
            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data['img'][0],
                                                  torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr)
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()



    print("ffff")



if __name__ == "__main__":
    print("Start")
    bevformer_infer()
    print("End")


