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
import json
from argparse import Namespace
import warnings

import training.training_loop as training_loop
# from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops


#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------
class DotDict(dict):
    """A dictionary with dot-access access. Keeps JSON serializability."""
    def __getattr__(self, name):
        val = self.get(name)
        if isinstance(val, dict):
            return DotDict(val)
        elif isinstance(val, list):
            return [DotDict(i) if isinstance(i, dict) else i for i in val]
        return val

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name] 

def load_config_as_easydict(path):
    def recursive_easydict(d):
        if isinstance(d, dict):
            return dnnlib.EasyDict({k: recursive_easydict(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [recursive_easydict(item) for item in d]
        else:
            return d

    with open(path, 'r') as f:
        config_dict = json.load(f)
    
    return recursive_easydict(config_dict)

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


    # with open("cucp_config.json" , "r") as f:
    #     config_kwargs = json.load(f)
      # args = DotDict(config_kwargs)
    args = load_config_as_easydict("cucp_config.json")
        
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
# Custom modules
from training.networks import VGG, module_no_grad
import legacy
from torch_utils import misc
import dnnlib
import numpy as np
import PIL
import matplotlib.pyplot as plt

def load_model(model_path,vgg_path,device):
    with open(model_path,'rb') as f:
        contents = legacy.load_network_pkl(f) # Pickles weights and source code

    # Get exponential movign average model
    G_ema = contents['G_ema']

    # Load DEX VGG classifier
    vgg = VGG()
    vgg_state_dict = torch.load(vgg_path)
    vgg_state_dict = {k.replace('-', '_'): v for k, v in vgg_state_dict.items()}
    vgg.load_state_dict(vgg_state_dict)
    module_no_grad(vgg) #!important

    # Set classifier
    G_ema.skip_grad_blur.model.classifier = vgg
    # No grad
    G_ema = G_ema.to(device).eval().requires_grad_(False)
    # No grad on VGG

    return G_ema

def run_model(G, img, label:torch.Tensor, global_blur_val=None, mask_blur_val=None, return_msk = False):
    # Tranform label to One Hot Encoding
    cls = torch.nn.functional.one_hot(
        torch.tensor(label.clip(20,65)),
        num_classes=G.attr_map.fc0.init_args[0]
    ).to(img.device)

    # Content encoder
    _,c_out_skip = G.content_enc(img)

    # Style encodder
    s_out = G.style_enc(img)[0].mean((2, 3))

    truncation_psi=1
    truncation_cutoff=None
    s_out = G.style_map(s_out, None, truncation_psi, truncation_cutoff)

    # age mapping
    a_out = G.attr_map(cls.to(s_out.device), None, truncation_psi, truncation_cutoff)

    # Style mapping and Age mapping are interleaved for the corresponding
    # weight demodulation modules
    w = G.__interleave_attr_style__(a_out, s_out)

    # Global blur
    for i,(f,_) in enumerate(zip(G.skip_transf, c_out_skip)):
        if f is not None:
            c_out_skip[i] = G._batch_blur(c_out_skip[i], blur_val = global_blur_val)

    # Masked blur
    cam = G.skip_grad_blur(img.float())
    msk = cam
    for i, (f, c) in enumerate(zip(G.skip_transf, c_out_skip)):
        if f is not None:
            im_size = c.size(-1)
            blur_c = G._batch_blur(c, blur_val= mask_blur_val)
            if msk.size(2) != im_size:
                msk = F.interpolate(msk,size=(im_size,im_size), mode='area')
            merged_c = c * msk + blur_c * (1 - msk)
            c_out_skip[i] = merged_c


    # Decoder
    img_out = G.image_dec(c_out_skip, w)

    if return_msk:
        to_return = (img_out,msk,cam) if G.learn_mask is not None else (img_out,None,None)
    else:
        to_return = img_out

    # assert(all(x.grad is None for x in G.parameters()))
    # assert(all(x.grad is None for x in G.skip_grad_blur.model.get_classifier().parameters()))
    # G.zero_grad()

    return to_return

# Transform tensor to uint8 image
def to_uint8(im_tensor):
    im_tensor = (im_tensor.detach().cpu().numpy().transpose((1,2,0))+1)*(256/2)
    im_tensor = np.clip(im_tensor,0,255).astype(np.uint8)
    return im_tensor

def test(weights_path , vgg_path , sample_images_path):
    weights_path = weights_path
    vgg_path = vgg_path
    # Images path
    sample_images_path= sample_images_path
    KEY = FFHQ_RR_KEY = "hrfae" # Model trained on HRFAE dataset
    configs = {
        FFHQ_RR_KEY: dict(
            gdrive_id="17BOTEa6z3r6JFVs1KDutDxWEkTWbzaeD",
            side=224,
            classes=(20,65))
    }

    # CUDA device
    device = torch.device('cuda',0)

    # Model GDrive ID
    model_id = configs[KEY]['gdrive_id']
    # Side of input images
    img_side = configs[KEY]['side']
    # Labels range for examples generation
    data_labels_range = configs[KEY]['classes']
    filenames_batch = [
    os.path.join(sample_images_path,f)
    for  f in next(iter(os.walk(sample_images_path)))[2]
    if f[-4:] == '.png'
    ]

    # loading the models weights
    G_ema = load_model(
    weights_path,
    vgg_path,
    device)

    # Image side
    side = configs[KEY]['side']
    # Read images
    imgs = [np.array(PIL.Image.open(f).resize((side,side)),dtype=np.float32).transpose((2,0,1)) for f in filenames_batch]
    # Transform to tensors
    im_in_tensor = (torch.tensor(np.array(imgs))/256*2-1).cuda() # Values {-1,1}

    # Aging steps
    steps = 5 # N steps
    # Repeat images N times
    n_images = im_in_tensor.shape[0]
    im_in_tensor_exp = im_in_tensor[:,None].expand([n_images,steps,*im_in_tensor.shape[1:]]).reshape([-1,*im_in_tensor.shape[1:]])
    # Define target ages
    labels_exp = torch.tensor(np.repeat(np.linspace(*data_labels_range,steps,dtype=int)[:,None],n_images,1).T.reshape(-1))

    batch_size = 12
    # Run model
    im_out_tensor_exp = torch.concat([run_model(
        G_ema,
        mini_im,
        mini_label,
        global_blur_val=0.2, # CUSP global blur
        mask_blur_val=0.8)   # CUSP masked blur
        for mini_im, mini_label
        in zip(
            im_in_tensor_exp.split(batch_size),
            labels_exp.split(batch_size)
        )])
    # Transform to [batch_size, N_ages, W, H , C]
    im_out_tensor = im_out_tensor_exp.reshape([-1,steps,*im_out_tensor_exp.shape[1:]])

    generated_path = "result"
    os.makedirs(generated_path , exist_ok=True)
    for f_name ,im_out , lables in zip(filenames_batch, im_out_tensor , labels_exp.numpy().reshape(-1,steps)):
        for img , l in zip(im_out, lables):
            l = (l  //10) * 10
            subdir_path = os.path.join(generated_path , str(l))
            print(subdir_path)
            os.makedirs(subdir_path , exist_ok=True)
            img = to_uint8(img)
            result_img = PIL.Image.fromarray(img)
            save_path = os.path.join(subdir_path , f_name.split("/")[-1])
            result_img.save(save_path)
            print(save_path)
        print("done!!!")


#----------------------------------------------------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    with open("config.json" , "r") as f:
        config_kwargs = json.load(f)
        args = Namespace(**config_kwargs)
    config_kwargs.pop('outdir', None)
    config_kwargs.pop('dry_run', None)
    if args.mode == "train":
        main(None ,args.outdir ,args.dry_run ,**config_kwargs) # pylint: disable=no-value-for-parameter
    
    # vgg_path
    vgg_path = None
    new_model_path=None 
    test_dataset_path = None
    test(vgg_path , new_model_path,test_dataset_path)
