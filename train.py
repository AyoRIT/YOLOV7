"""
File: train.py
Description:
    This script is a core training module for a YOLO-based object detection model. It handles 
    data preparation, model initialization, loss computation, and optimization across 
    distributed or single-node systems. Additional features include logging, automatic anchor 
    adjustment, and integration with TensorBoard and W&B for tracking experiments.

Author: Daniel Gebura
Date: 12/5/2024
"""

# -----------------------------
# Import Required Libraries
# -----------------------------

import argparse
import logging
import math
import os
from pathlib import Path
import random
import time
from copy import deepcopy
from threading import Thread
import numpy as np
import yaml

# torch: Core library for building and training deep learning models
import torch.distributed as dist  # Manages distributed training across multiple devices
import torch.nn as nn  # Provides modules and layers for constructing models
import torch.nn.functional as F  # Contains functions for operations on tensors
import torch.optim as optim  # Optimization algorithms for training
import torch.optim.lr_scheduler as lr_scheduler  # Learning rate scheduling during training
import torch.utils.data  # Utilities for data loading and batching
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Local modules and utilities
import test  # Imports test.py for computing mean average precision (mAP) after each epoch
from models.experimental import attempt_load  # For loading experimental model architectures
from models.yolo import Model  # YOLO model definition
from utils.autoanchor import check_anchors  # Checks and adjusts anchor sizes for datasets
from utils.datasets import create_dataloader  # Data loading utilities for training and validation
from utils.general import (  # General utilities for various training tasks
    labels_to_class_weights,
    increment_path,
    labels_to_image_weights,
    init_seeds,
    fitness,
    strip_optimizer,
    get_latest_run,
    check_dataset,
    check_file,
    check_git_status,
    check_img_size,
    check_requirements,
    print_mutation,
    set_logging,
    one_cycle,
    colorstr
)
from utils.google_utils import attempt_download  # Download utilities for fetching models or data
from utils.loss import ComputeLoss, ComputeLossOTA  # Loss computation for YOLO
from utils.plots import (  # Functions for creating visualizations like images and training metrics
    plot_images,
    plot_labels,
    plot_results,
    plot_evolution
)
from utils.torch_utils import (  # Torch-specific helper functions
    ModelEMA,
    select_device,
    intersect_dicts,
    torch_distributed_zero_first,
    is_parallel
)
from utils.wandb_logging.wandb_utils import (  # Utilities for logging to Weights & Biases
    WandbLogger,
    check_wandb_resume
)

# -----------------------------
# Global Logger Initialization
# -----------------------------

# Configures the global logger for the training script
logger = logging.getLogger(__name__)


def train(hyp, opt, device, tb_writer=None):
    """
    Train the YOLO model with the given hyperparameters and options.

    Args:
        hyp (dict): Hyperparameter dictionary containing training configurations such as learning rate, momentum, etc.
        opt (argparse.Namespace): Parsed command-line arguments including data paths, batch sizes, device settings, etc.
        device (torch.device): Device (CPU or GPU) to use for training.
        tb_writer (SummaryWriter, optional): TensorBoard writer instance for logging metrics. Defaults to None.

    Returns:
        tuple: Final training results, including precision, recall, mAP metrics, and losses.
    """
    # Log hyperparameters
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    
    # Parse options and paths
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # Directories setup
    wdir = save_dir / 'weights'  # Weight files directory
    wdir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    last = wdir / 'last.pt'  # Path for saving the last checkpoint
    best = wdir / 'best.pt'  # Path for saving the best model
    results_file = save_dir / 'results.txt'  # Path for training results

    # Save configuration files for reproducibility
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)  # Save hyperparameters
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)  # Save options

    # Configuration settings
    plots = not opt.evolve  # Whether to create plots during training
    cuda = device.type != 'cpu'  # Check if CUDA is available
    init_seeds(2 + rank)  # Set random seeds for reproducibility

    # Load data configuration
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # Parse YAML data configuration
    is_coco = opt.data.endswith('coco.yaml')  # Check if the dataset is COCO

    # Logging setup
    loggers = {'wandb': None}  # Initialize loggers
    if rank in [-1, 0]:  # Main process
        opt.hyp = hyp  # Attach hyperparameters to options
        # Attempt to resume training from weights
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb  # Set up W&B logging
        data_dict = wandb_logger.data_dict  # Update data dictionary if modified by W&B

        # Update training configuration if resuming from W&B
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

    # Number of classes
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # Single-class or multi-class
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # Class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {opt.data}'

    # Model initialization
    pretrained = weights.endswith('.pt')  # Check if loading a pretrained model
    if pretrained:
        # Handle pretrained weights
        with torch_distributed_zero_first(rank):  # Ensure synchronized downloading
            attempt_download(weights)
        ckpt = torch.load(weights, map_location=device)  # Load checkpoint
        # Create model with hyperparameters
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        # Load pretrained weights into model
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []
        state_dict = intersect_dicts(ckpt['model'].float().state_dict(), model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f'Transferred {len(state_dict)}/{len(model.state_dict())} items from {weights}')
    else:
        # Initialize a new model
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

    # Ensure the dataset is valid
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)
    train_path = data_dict['train']  # Training data path
    test_path = data_dict['val']  # Validation data path

    # Freezing specified layers
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True  # Enable training for all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')  # Freeze specific layers
            v.requires_grad = False

    # Initialize optimizer
    nbs = 64  # Nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # Accumulate loss for larger batches
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # Adjust weight decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # Create parameter groups for the optimizer
    pg0, pg1, pg2 = [], [], []  # Separate groups for biases, BatchNorm weights, and other weights
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # Add biases to pg2
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # Add BatchNorm weights to pg0 (no decay)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # Add other weights to pg1 (apply decay)

        # Handle specific layers with implicit parameters
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)

        # Handle attention layers
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):
                pg0.append(v.attn.relative_position_bias_table)

        # Handle reparameterized bottleneck layers
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):
                pg0.append(v.rbr_dense.vector)

    # Select optimizer (Adam or SGD) and configure its parameters
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # Add other parameter groups to the optimizer
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # Add pg1 (with weight decay)
    optimizer.add_param_group({'params': pg2})  # Add pg2 (biases without weight decay)
    logger.info(f'Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other')

    # Learning rate scheduler setup
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # Linear scheduler
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # Cosine scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Exponential Moving Average (EMA) for model weights
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume training from a checkpoint if provided
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])  # Load optimizer state
            best_fitness = ckpt['best_fitness']  # Restore best fitness value

        # Resume EMA state
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Restore results and epochs
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # Restore training results
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, f"{weights} training to {epochs} epochs is finished, nothing to resume."

        if epochs < start_epoch:
            logger.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} additional epochs.")
            epochs += ckpt['epoch']

    # Set image sizes and grid sizes
    gs = max(int(model.stride.max()), 32)  # Maximum stride for the model
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # Ensure image sizes are multiples of gs

    # Enable multi-GPU training (DataParallel mode) if applicable
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Enable synchronized BatchNorm if specified
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')


    # Create dataloaders for training and validation
    dataloader, dataset = create_dataloader(
        train_path, imgsz, batch_size, gs, opt,
        hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
        world_size=opt.world_size, workers=opt.workers,
        image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: ')
    )
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # Maximum label class
    nb = len(dataloader)  # Number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {opt.data}. Possible class labels are 0-{nc - 1}'

    # Set up validation dataloader for process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(
            test_path, imgsz_test, batch_size * 2, gs, opt,
            hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
            world_size=opt.world_size, workers=opt.workers,
            pad=0.5, prefix=colorstr('val: ')
        )[0]

        # Initialize plotting and labels if not resuming
        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)  # Collect all labels
            c = torch.tensor(labels[:, 0])  # Extract class labels
            if plots:
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)  # Log class histogram to TensorBoard

            # Check and adjust anchors if necessary
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # Pre-reduce anchor precision

    # Enable DistributedDataParallel (DDP) mode if applicable
    if cuda and rank != -1:
        model = DDP(
            model, device_ids=[opt.local_rank], output_device=opt.local_rank,
            find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules())
        )

    # Adjust hyperparameters for multi-scale and multiple layers
    hyp['box'] *= 3. / model.model[-1].nl  # Scale box loss by number of layers
    hyp['cls'] *= nc / 80. * 3. / model.model[-1].nl  # Scale class loss by number of classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / model.model[-1].nl  # Scale object loss by image size and layers
    hyp['label_smoothing'] = opt.label_smoothing  # Apply label smoothing
    model.nc = nc  # Attach number of classes to the model
    model.hyp = hyp  # Attach hyperparameters to the model
    model.gr = 1.0  # IoU loss ratio (obj_loss = 1.0 or IoU)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # Attach class weights
    model.names = names  # Attach class names

    # Start the training process
    t0 = time.time()  # Record start time
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # Number of warmup iterations
    maps = np.zeros(nc)  # Initialize mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # Initialize training results
    scheduler.last_epoch = start_epoch - 1  # Sync scheduler with the start epoch
    scaler = amp.GradScaler(enabled=cuda)  # Initialize gradient scaler for AMP
    compute_loss_ota = ComputeLossOTA(model)  # Initialize loss computation for OTA
    compute_loss = ComputeLoss(model)  # Initialize standard loss computation
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # Save initial model checkpoint
    torch.save(model, wdir / 'init.pt')

    # Epoch loop
    for epoch in range(start_epoch, epochs):
        model.train()  # Set model to training mode

        # Update image weights for class balancing if enabled
        if opt.image_weights:
            if rank in [-1, 0]:  # Main process
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # Compute class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # Compute image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # Weighted random sampling
            if rank != -1:  # Broadcast indices for DDP
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)  # Initialize mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)  # Update sampler for DDP
        pbar = enumerate(dataloader)  # Initialize progress bar
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))

        # Display progress bar for the main process
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)

        optimizer.zero_grad()  # Reset gradients

        for i, (imgs, targets, paths, _) in pbar:  # Iterate over batches
            ni = i + nb * epoch  # Number of integrated batches
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # Normalize images to [0, 1]

            # Warmup phase for dynamic hyperparameters
            if ni <= nw:
                xi = [0, nw]  # Warmup interval
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale training: dynamically resize images during training
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # Random size in range
                sf = sz / max(imgs.shape[2:])  # Scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # New shape (adjusted to grid size)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)  # Resize images

            # Forward pass through the model
            with amp.autocast(enabled=cuda):  # Automatic Mixed Precision (AMP) for efficiency
                pred = model(imgs)  # Get predictions
                # Compute loss
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))

                # Scale loss for distributed training or quad learning
                if rank != -1:
                    loss *= opt.world_size
                if opt.quad:
                    loss *= 4.

            # Backward pass and gradient accumulation
            scaler.scale(loss).backward()  # Backpropagation

            # Optimize when accumulated gradients reach the threshold
            if ni % accumulate == 0:
                scaler.step(optimizer)  # Update model parameters
                scaler.update()  # Update the scaler for AMP
                optimizer.zero_grad()  # Reset gradients
                if ema:
                    ema.update(model)  # Update EMA weights

            # Logging and displaying progress
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # Update mean loss
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # GPU memory
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1])  # Format string
                pbar.set_description(s)  # Update progress bar description

                # Plot training samples
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # File path for plots
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({
                        "Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name)
                                    for x in save_dir.glob('train*.jpg') if x.exists()]
                    })

        # Update learning rate scheduler after every epoch
        lr = [x['lr'] for x in optimizer.param_groups]  # Get current learning rates
        scheduler.step()  # Update learning rates

        # Validate the model (process 0 or single GPU)
        if rank in [-1, 0]:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs  # Check if this is the last epoch

            # Perform validation and compute metrics if necessary
            if not opt.notest or final_epoch:
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(
                    data_dict,
                    batch_size=batch_size * 2,
                    imgsz=imgsz_test,
                    model=ema.ema,
                    single_cls=opt.single_cls,
                    dataloader=testloader,
                    save_dir=save_dir,
                    verbose=nc < 50 and final_epoch,
                    plots=plots and final_epoch,
                    wandb_logger=wandb_logger,
                    compute_loss=compute_loss,
                    is_coco=is_coco,
                    v5_metric=opt.v5_metric
                )

                # Log validation metrics to TensorBoard
                if tb_writer:
                    tb_writer.add_scalar("Validation/Loss", results[4] + results[5] + results[6], epoch)
                    tb_writer.add_scalar("Validation/Accuracy", results[0], epoch)  # Precision as a proxy for accuracy
                    tb_writer.add_scalar("Validation/Precision", results[0], epoch)
                    tb_writer.add_scalar("Validation/Recall", results[1], epoch)
                    tb_writer.add_scalar("Validation/mAP_0.5", results[2], epoch)
                    tb_writer.add_scalar("Validation/mAP_0.5_0.95", results[3], epoch)

            # Save results to file
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # Append metrics and validation loss

            # Update best fitness based on validation results
            fi = fitness(np.array(results).reshape(1, -1))  # Compute weighted combination of metrics
            if fi > best_fitness:
                best_fitness = fi  # Update best fitness score
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model checkpoints
            if (not opt.nosave) or (final_epoch and not opt.evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(model.module if is_parallel(model) else model).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None
                }

                # Save last and best model checkpoints
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)

                # Save additional checkpoints at specific intervals
                if epoch == 0:
                    torch.save(ckpt, wdir / f'epoch_{epoch:03d}.pt')
                elif (epoch + 1) % 25 == 0 or epoch >= (epochs - 5):
                    torch.save(ckpt, wdir / f'epoch_{epoch:03d}.pt')
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(last.parent, opt, epoch, fi, best_model=best_fitness == fi)

                del ckpt  # Free memory

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')
