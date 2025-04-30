# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""

import os
import click
import re
import json
import tempfile
import torch
import dnnlib

import training.training_loop as training_loop
# from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops


#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_loop_kwargs(
    # General options (not included in desc).
    gpus       = None, # GPUs: [<int>], default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks
    seed       = None, # Random seed: <int>, default = 0

    # Dataset.
    dataset = None, #['ffhq_aug','ffhq_lat']
    cmap_kind = None, # ["identity",'gauss','bins','number']
    age_np  = None, # [[img_name, age]] (required): <path>
    age_np_test= None, # [[img_name, age]] (not required)
    csv = None,
    data       = None, # Training dataset (required): <path>
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    classifier_path = None,
    mask_path = None,

    # Base config.
    cfg        = None, # Base config: 'auto' (default), '224', '256'
    gamma      = None, # Override R1 gamma: <float>
    kimg       = None, # Override training duration: <int>
    batch      = None, # Override batch size: <int>

    age_loss   = None, # ('ce', 'mvl')
    downsamples= None, # Number of downsamples
    bias       = None, # Use bias in convolutional synthesis layers
    class_w    = None, # Classification weight
    cycle_w = None,
    skip_layers= None,
    skip_kind  = None, # 'linear' or 'cbam'
    age_margin = None,
    rgb_attention = None,
    rgb_reg = None,
    soft_margin = None,
    blur_skip = None,
    blur_msk = None,
    act_reg = None, # None, 'l1' or 'l2'
    skip_grad_blur = None,
    learn_mask = None,
    mixing_prob = None,
    disc_class = None,
    fake_rec = None,
    style_enc = None,
    bottleneck_class = None,
    finetune = None,
    class_kind = None,

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers

    # Performance options (not included in desc).
    fp32       = None, # Disable mixed-precision training: <bool>, default = False
    nhwc       = None, # Use NHWC memory format with FP16: <bool>, default = False
    allow_tf32 = None, # Allow PyTorch to use TF32 for matmul and convolutions: <bool>, default = False
    nobench    = None, # Disable cuDNN benchmarking: <bool>, default = False
    workers    = None, # Override number of DataLoader workers: <int>, default = 3
):
    args = dnnlib.EasyDict()

    args.gb = 'Fixed'
    args.msk = 'Reversed'

    # ------------------------------------------
    # General options: gpus, snap, metrics, seed
    # ------------------------------------------
    gpu_list = gpus
    gpus = len(gpus)

    # Prevent memory pinning on GPU:0
    torch.cuda.set_device(torch.device('cuda',int(gpu_list[0])))

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus
    args.gpu_list = gpu_list

    if snap is None:
        snap = 20
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap // 12
    args.network_snapshot_ticks = snap

    if seed is None:
        seed = 0
    assert isinstance(seed, int)
    args.random_seed = seed

    if age_loss is None:
        age_loss = 'ce'

    if downsamples is None:
        downsamples = 8

    if bias is None:
        bias = True

    if class_w is None:
        class_w = 0.1

    if skip_layers is None:
        skip_layers = []
    skip_layers = [int(x) for x in skip_layers]

    if skip_kind is None:
        skip_kind = 'linear'

    if age_margin is None:
        age_margin = 20

    if rgb_attention is None:
        rgb_attention = False

    if rgb_reg.lower() == "none":
        rgb_reg = None

    if soft_margin is None:
        soft_margin = False

    if blur_skip is None:
        blur_skip = False

    if act_reg.lower() == "none":
        act_reg = None

    if skip_grad_blur.lower() == "none":
        skip_grad_blur = None

    if learn_mask.lower() == "none":
        learn_mask = None

    if mixing_prob is None:
        mixing_prob = 0.

    if disc_class is None:
        disc_class = False

    if fake_rec is None:
        fake_rec = False

    if cycle_w is None:
        cycle_w = 0.

    if blur_msk is None:
        blur_msk = "fixed"

    if style_enc is None:
        style_enc = True

    if bottleneck_class is None:
        bottleneck_class = False

    if class_kind is None:
        class_kind = 'all'


    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------

    assert (age_np is not None or csv is not None)
    # assert isinstance(age_np, str)
    assert data is not None
    assert isinstance(data, str)
    assert classifier_path is not None
    assert isinstance(classifier_path, str)
    # ['ffhq_aug','celeba','bdd100k','afhq']
    if dataset == 'ffhq_aug':
        args.training_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.AgeDataset', age_np_path=age_np,
                                                   image_path = data, max_size=None, xflip=False, cmap_pre_kind=cmap_kind)
        args.test_set_kwargs = dnnlib.EasyDict(class_name='training.dataset.AgeDataset', age_np_path=age_np_test,
                                                   image_path=data, max_size=None, xflip=False, cmap_pre_kind=cmap_kind) \
                                                if age_np_test is not None else None
    else:
        args.training_set_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageCSVDataset', csv_path = csv, image_path=data, is_train=True,
            cmap_pre_kind=cmap_kind)
        args.test_set_kwargs = dnnlib.EasyDict(
            class_name='training.dataset.ImageCSVDataset', csv_path=csv, image_path=data, is_train=False,
            cmap_pre_kind=cmap_kind)

        args.training_set_kwargs.transforms = dataset
        args.test_set_kwargs.transforms = dataset

    args.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=3, prefetch_factor=2)
    try:
        training_set = dnnlib.util.construct_class_by_name(**args.training_set_kwargs) # subclass of training.dataset.Dataset
        args.training_set_kwargs.resolution = training_set.resolution # be explicit about resolution
        args.training_set_kwargs.use_labels = training_set.has_labels # be explicit about labels
        args.training_set_kwargs.max_size = len(training_set) # be explicit about dataset size
        desc = training_set.name
        label_dim  = training_set.label_dim
        del training_set # conserve memory
    except IOError as err:
        raise UserError(f'--data: {err}')


    if mirror is None:
        mirror = False
    assert isinstance(mirror, bool)
    if mirror:
        desc += '-mirror'
        args.training_set_kwargs.xflip = True

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(
            ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05,
            style_layers=8, style_dim = 512, attr_layers=8, n_downsample = downsamples
        ), # Populated dynamically based on resolution and GPU count.
        '224':  dict(
            ref_gpus=1,  kimg=25000,  mb=14, mbstd=4,  fmaps=0.5, lrate=0.0025, gamma=0.56,    ema=5,  ramp=None,
            style_layers=8, style_dim = 512, attr_layers=8, n_downsample = downsamples
        ),
        '256': dict(
            ref_gpus=1, kimg=25000, mb=14, mbstd=4, fmaps=0.5, lrate=0.0025, gamma=0.56, ema=5, ramp=None,
            style_layers=8, style_dim=512, attr_layers=8, n_downsample=downsamples
        ),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    res = args.training_set_kwargs.resolution
    # chn = training_set.num_channels
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mb = 3
        # spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32


    args.G_kwargs = dnnlib.EasyDict(
        class_name='training.networks.Image2Image', n_downsample=spec.n_downsample, mapping_kwargs=dnnlib.EasyDict(),
        skip_layers = skip_layers, skip_kind = skip_kind, synthesis_kwargs=dnnlib.EasyDict(), blur_skip=blur_skip,
        skip_grad_blur=skip_grad_blur,  learn_mask = learn_mask, attr_dim = label_dim, blur_msk=blur_msk,
        style_enc = style_enc, bottleneck_class=bottleneck_class, finetune= finetune
    )
    args.D_kwargs = dnnlib.EasyDict(
        class_name='training.networks.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(),
        epilogue_kwargs=dnnlib.EasyDict(), finetune= finetune)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = 512
    args.G_kwargs.synthesis_kwargs.bias = bias
    args.G_kwargs.style_layers = spec.style_layers
    args.G_kwargs.style_dim = spec.style_dim
    args.G_kwargs.attr_layers = spec.attr_layers
    args.G_kwargs.synthesis_kwargs.rgb_attention = rgb_attention
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.D_kwargs.epilogue_kwargs.mbstd_num_channels = 0

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)

    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=spec.lrate, betas=[0,0.99], eps=1e-8)


    loss_weight = dnnlib.EasyDict(
                                       rec = 10 if not finetune else 0,
                                       cla = class_w if not finetune else 0,#0.06,
                                       adv = 1,
                                       rgb = 1 if not finetune else 0,
                                       act = 1 if not finetune else 0,
                                       msk = 0.1 if not finetune else 0,
                                       msk_smooth = 1e-5 if not finetune else 0,
                                       cam = 10 if not finetune else 0,
                                       fre = (2. if fake_rec else 0.) if not finetune else 0 ,
                                       cycle = cycle_w
                                   )
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.I2ILoss', r1_gamma=spec.gamma,
                                       loss_weights = loss_weight,
                                       rgb_reg= rgb_reg,
                                       act_reg = act_reg,
                                       mixing_probability = mixing_prob,
                                       disc_class = disc_class)




    vgg_path = 'dex_imdb_wiki.caffemodel.pt'
    if dataset == 'ffhq_aug':
        classifier_kwargs = dnnlib.EasyDict(class_name='training.loss.DEXAgeClassifier',
                                                vgg_path=classifier_path, own_relu=True, outclass=class_kind)
        guidedback_kwargs = classifier_kwargs
    elif dataset in ['bdd100k']:
        classifier_kwargs = dnnlib.EasyDict(class_name='training.loss.ResNetClassifier',
                                                             model_path=classifier_path, mask_path=mask_path,
                                                             outclass='one')
        guidedback_kwargs = classifier_kwargs
    elif dataset in ['ffhq_lat']:
        classifier_kwargs = dnnlib.EasyDict(class_name='training.loss.ResNetClassifier',
                                                             model_path=classifier_path, mask_path=mask_path,
                                                             outclass='one')
        guidedback_kwargs = dnnlib.EasyDict(class_name='training.loss.DEXAgeClassifier',
                                                             vgg_path=vgg_path, own_relu=True)
    elif dataset in ['zebra','afhq']:
        classifier_kwargs = dnnlib.EasyDict(class_name='training.loss.VGGBNClassifier',
                                                             model_path=classifier_path, outclass='one')
        guidedback_kwargs = classifier_kwargs
    else:
        raise NotImplementedError

    args.loss_kwargs.classifier_kwargs = classifier_kwargs
    args.G_kwargs.classifier_kwargs = guidedback_kwargs

    if dataset == 'ffhq_aug':
        if soft_margin:
            args.loss_kwargs.random_class_kwargs = dnnlib.EasyDict(class_name='training.loss.SoftMarginRandomAge',
                                                                   age_min=20, age_max=70)
        else:
            # args.loss_kwargs.random_class_kwargs = dnnlib.EasyDict(class_name='training.loss.HardRandomAge',
            #                                                        age_margin=age_margin, age_min=20, age_max=70)
            args.loss_kwargs.random_class_kwargs = dnnlib.EasyDict(class_name='training.loss.RandomAge',
                                                                   age_min=20, age_max=70)
    elif dataset in ['bdd100k', 'afhq', 'zebra','ffhq_lat']:
        args.loss_kwargs.random_class_kwargs = dnnlib.EasyDict(class_name='training.loss.ChangeAll')
    else:
        raise NotImplementedError

    if dataset == 'ffhq_aug':
        args.loss_kwargs.classification_loss_kwargs = dnnlib.EasyDict(class_name='training.loss.MeanVarLoss',
                                                             reduction='mean')
    elif dataset in ['celeba', 'bdd100k', 'afhq', 'zebra','ffhq_lat']:
        args.loss_kwargs.classification_loss_kwargs = dnnlib.EasyDict(class_name='torch.nn.BCEWithLogitsLoss',
                                                             reduction='mean')
    else:
        raise NotImplementedError

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp


    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_kwargs.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        desc += f'-batch{batch}'
        args.batch_size = batch
        args.batch_gpu = batch // gpus

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32 is None:
        fp32 = False
    assert isinstance(fp32, bool)
    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc is None:
        nhwc = False
    assert isinstance(nhwc, bool)
    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True

    if nobench is None:
        nobench = False
    assert isinstance(nobench, bool)
    if nobench:
        args.cudnn_benchmark = False

    if allow_tf32 is None:
        allow_tf32 = False
    assert isinstance(allow_tf32, bool)
    if allow_tf32:
        args.allow_tf32 = True

    if workers is not None:
        assert isinstance(workers, int)
        if not workers >= 1:
            raise UserError('--workers must be at least 1')
        args.data_loader_kwargs.num_workers = workers

    return desc, args

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', args.gpu_list[rank]) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    # if rank != 0:
    #     custom_ops.verbosity = 'none'
    custom_ops.verbosity = 'none'

    # Execute training loop.
    gpu_id = args.gpu_list[rank]
    del args.gpu_list
    training_loop.training_loop(rank=rank, gpu_id=gpu_id, **args)

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# General options.
@click.option('--outdir', help='Where to save the results', required=True, metavar='DIR')
@click.option('--gpus', help='GPUs to use [default: 0]', type=CommaSeparatedList())
@click.option('--snap', help='Snapshot interval [default: 50 ticks]', type=int, metavar='INT')
# @click.option('--metrics', help='Comma-separated list or "none" [default: fid50k_full]', type=CommaSeparatedList())
@click.option('--seed', help='Random seed [default: 0]', type=int, metavar='INT')
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)

# Dataset.
@click.option('--dataset', help='Dataset', required=True, type=click.Choice(['ffhq_aug','ffhq_lat','celeba','bdd100k','afhq','zebra']))
@click.option('--cmap_kind', help='Discriminator class mapping', type=click.Choice(["identity",'gauss','bins','number']))
@click.option('--age_np', help='[[img_path, age]] numpy', metavar='DIR')
@click.option('--age_np_test', help='[[img_path, age]] numpy', metavar='DIR')
@click.option('--csv', help='[[img_path, age]] numpy', metavar='DIR')
@click.option('--data', help='Training data (directory or zip)', metavar='PATH', required=True)
@click.option('--mirror', help='Enable dataset x-flips [default: false]', type=bool, metavar='BOOL')

# Discriminator
@click.option('--classifier_path', help='Classifier path', required=True)
@click.option('--mask_path', help='Classifier mask path')

# Base config.
@click.option('--cfg', help='Base config [default: auto]', type=click.Choice(['auto', '224','256']))
@click.option('--gamma', help='Override R1 gamma', type=float)
@click.option('--kimg', help='Override training duration', type=int, metavar='INT')
@click.option('--batch', help='Override batch size', type=int, metavar='INT')

# @click.option('--age_loss', help='Age loss [default: CrossEntropy]', type=click.Choice(['ce', 'mvl']))
@click.option('--downsamples', help='Number of generator downsamples (default 8)', type=int, metavar='INT')
@click.option('--bias', help='Use bias in Synthesis blocks [default: true]', type=bool, metavar='BOOL')
@click.option('--class_w', help='Classification weight [default: 0.1]', type=float, metavar='FLOAT')
@click.option('--cycle_w', help='Cycle consistency weight [default: 0.0]', type=float, metavar='FLOAT')
@click.option('--skip_layers', help='Layers for skip connect', type=CommaSeparatedList())
@click.option('--skip_kind', help='Skip connection modifier (cbam/linear) [default: "linear"]',
              type=click.Choice(['cbam', 'linear']))
@click.option('--age_margin', help='Margin from predicted for generator random age sampling', type=int, metavar='INT')
@click.option('--rgb_attention', help='Add attention on RGB [default: false]', type=bool, metavar='BOOL')
@click.option('--rgb_reg', help='RGB output regularization strategy ["none","std2","margin"]',
              type=click.Choice(['none','std2', 'margin']), default="none")
@click.option('--soft_margin', help='Soft margin [default: false]', type=bool, metavar='BOOL')
@click.option('--blur_skip', help='Train with random skip connection blur [default: false]', type=bool, metavar='BOOL')
@click.option('--blur_msk', help='Masked skip connection blur ["fixed","random"] [default: "fixed"]',
              type=click.Choice(["fixed","random"]), default="fixed")
@click.option('--act_reg', help='Activation regularization strategy ["none","l1","l2"]',
              type=click.Choice(['none','l1', 'l2']), default="none")
@click.option('--skip_grad_blur', help='Masked skip connection blur ["none","gb","camgb"]',
              type=click.Choice(["none","gb","camgb"]), default="none")
@click.option('--learn_mask', help='Learn masking ["none","skip","skip_cam","rgb","rgb_cam"]',
              type=click.Choice(["none"]+[f'{k}_{sufix}'
                                          for k in ["skip","skip_cam","rgb","rgb_cam"]
                                          for sufix in ['','random']]), default="none")
@click.option('--mixing_prob', help='Probability of style mixing [0,1] [default: 0.]', type=float)
@click.option('--disc_class', help='Pass class to StyleGAN discriminator', type=bool, metavar='BOOL')
@click.option('--fake_rec', help='Blurred reconstruction on fake images', type=bool, metavar='BOOL')
@click.option('--style_enc', help='Train separate style encoder [default: True]', type=bool, metavar='BOOL')
@click.option('--bottleneck_class', help='Class only on bottleneck [default: False]', type=bool, metavar='BOOL')
@click.option('--finetune', help='Finetune last n blocks [default: None]', type=int, metavar='BOOL')
@click.option('--class_kind', help='Classifier output seleccion (all/max) [default: "all"]',
              type=click.Choice(['all', 'max']))


# Transfer learning.
@click.option('--resume', help='Resume training [default: noresume]', metavar='PKL')
@click.option('--freezed', help='Freeze-D [default: 0 layers]', type=int, metavar='INT')

# Performance options.
@click.option('--fp32', help='Disable mixed-precision training', type=bool, metavar='BOOL')
@click.option('--nhwc', help='Use NHWC memory format with FP16', type=bool, metavar='BOOL')
@click.option('--nobench', help='Disable cuDNN benchmarking', type=bool, metavar='BOOL')
@click.option('--allow-tf32', help='Allow PyTorch to use TF32 internally', type=bool, metavar='BOOL')
@click.option('--workers', help='Override number of DataLoader workers', type=int, metavar='INT')

def main(ctx, outdir, dry_run, **config_kwargs):
    dnnlib.util.Logger(should_flush=True)

    # Setup training options.
    try:
        run_desc, args = setup_training_loop_kwargs(**config_kwargs)
    except UserError as err:
        ctx.fail(err)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training data:      {args.training_set_kwargs.image_path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    print(f'Number of images:   {args.training_set_kwargs.max_size}')
    print(f'Image resolution:   {args.training_set_kwargs.resolution}')
    print(f'Conditional model:  {args.training_set_kwargs.use_labels}')
    # print(f'Dataset x-flips:    {args.training_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
