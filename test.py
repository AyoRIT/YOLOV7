"""
File: test.py
Description:
    This script is designed for testing YOLO-based object detection models. It evaluates model performance
    by computing metrics such as precision, recall, and mAP, and supports options like saving predictions,
    hybrid auto-labeling, and generating COCO-compatible JSON results.

Author: Daniel Gebura
Date: 12/5/2024
"""
# -----------------------------
# Import Required Libraries
# -----------------------------
import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Local utilities for model loading, data handling, and evaluation
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import (coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements,
                           box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging,
                           increment_path, colorstr)
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=True,
         trace=False,
         is_coco=False,
         v5_metric=False):
    """
    Evaluate the YOLO model's performance on a dataset.

    Args:
        data (str or dict): Path to data YAML file or a dictionary with dataset info.
        weights (str or list, optional): Path to the model weights file(s). Defaults to None.
        batch_size (int): Batch size for inference. Defaults to 32.
        imgsz (int): Image size for inference. Defaults to 640.
        conf_thres (float): Confidence threshold for object detection. Defaults to 0.001.
        iou_thres (float): IoU threshold for non-maximum suppression (NMS). Defaults to 0.6.
        save_json (bool): Save predictions in COCO JSON format. Defaults to False.
        single_cls (bool): Treat the dataset as single-class. Defaults to False.
        augment (bool): Use augmented inference. Defaults to False.
        verbose (bool): Print detailed class-wise metrics. Defaults to False.
        model (nn.Module, optional): Preloaded model for inference. Defaults to None.
        dataloader (DataLoader, optional): Preloaded dataloader. Defaults to None.
        save_dir (Path): Directory to save results and images. Defaults to Path('').
        save_txt (bool): Save results to text files. Defaults to False.
        save_hybrid (bool): Save hybrid labels (predictions + ground truth). Defaults to False.
        save_conf (bool): Save confidence scores in text files. Defaults to False.
        plots (bool): Generate visualizations for predictions and results. Defaults to True.
        wandb_logger (WandbLogger, optional): W&B logger for experiment tracking. Defaults to None.
        compute_loss (function, optional): Function to compute training loss. Defaults to None.
        half_precision (bool): Use half-precision inference on CUDA. Defaults to True.
        trace (bool): Use TorchScript tracing for the model. Defaults to False.
        is_coco (bool): Indicate if the dataset is COCO. Defaults to False.
        v5_metric (bool): Use YOLOv5-style AP metric calculation. Defaults to False.

    Returns:
        tuple: Precision, recall, mAP@0.5, mAP@0.5:0.95, loss, and inference timings.
    """
    # -----------------------------
    # Model and Device Setup
    # -----------------------------
    training = model is not None  # Check if called during training
    if training:  # If called from train.py
        device = next(model.parameters()).device  # Get the device the model is on
    else:  # Called directly
        set_logging()  # Initialize logging
        device = select_device(opt.device, batch_size=batch_size)  # Select device

        # Configure directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # Increment run directory
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Create necessary dirs

        # Load the model
        model = attempt_load(weights, map_location=device)  # Load FP32 model
        gs = max(int(model.stride.max()), 32)  # Determine grid size
        imgsz = check_img_size(imgsz, s=gs)  # Ensure image size is a multiple of the grid size

        # Trace the model if specified
        if trace:
            model = TracedModel(model, device, imgsz)

    # Set half precision mode
    half = device.type != 'cpu' and half_precision  # Enable half-precision if on CUDA
    if half:
        model.half()

    # -----------------------------
    # Dataset and Dataloader Setup
    # -----------------------------
    model.eval()  # Set model to evaluation mode
    if isinstance(data, str):  # If data is a path to a YAML file
        is_coco = data.endswith('coco.yaml')  # Check if the dataset is COCO
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)  # Load dataset config from YAML
    check_dataset(data)  # Verify dataset paths

    # Number of classes
    nc = 1 if single_cls else int(data['nc'])

    # IoU thresholds for AP calculation
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    # Setup W&B logging parameters
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)

    # Create dataloader if not training
    if not training:
        if device.type != 'cpu':
            # Run a dummy forward pass to initialize model
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # Select data split
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    # Check if YOLOv5-style metrics are used
    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    # -----------------------------
    # Metrics and Results Initialization
    # -----------------------------
    seen = 0  # Number of images processed
    confusion_matrix = ConfusionMatrix(nc=nc)  # Initialize confusion matrix
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}  # Class names
    coco91class = coco80_to_coco91_class()  # Map COCO80 to COCO91 class indices

    # Print header for metrics
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)  # Initialize loss (box, objectness, class)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []  # Initialize lists for evaluation

    # -----------------------------
    # Batch Processing Loop
    # -----------------------------
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # Prepare images
        img = img.to(device, non_blocking=True)  # Move images to device
        img = img.half() if half else img.float()  # Convert to half-precision if applicable
        img /= 255.0  # Normalize pixel values to [0, 1]
        targets = targets.to(device)  # Move targets to device
        nb, _, height, width = img.shape  # Get image batch dimensions

        # Run model inference
        with torch.no_grad():
            t = time_synchronized()  # Start timing
            out, train_out = model(img, augment=augment)  # Get predictions and training outputs
            t0 += time_synchronized() - t  # Update inference time

            # Compute training loss if required
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # Add box, obj, cls losses

            # Perform Non-Maximum Suppression (NMS)
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # Convert to pixel values
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # Hybrid labels
            t = time_synchronized()  # Start timing
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t  # Update NMS time

        # Process predictions and ground truth
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]  # Ground truth labels for the current image
            nl = len(labels)  # Number of labels
            tcls = labels[:, 0].tolist() if nl else []  # Target classes
            path = Path(paths[si])  # Image path
            seen += 1  # Increment the count of processed images

            if len(pred) == 0:  # No predictions
                if nl:  # If ground truth exists
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Scale predictions to original image dimensions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # Scale to native-space

            # Save predictions to text file
            if save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # Normalization gain (whwh)
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Normalize to xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # Format label
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging for bounding boxes
            if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:
                if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
                    box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                                 "class_id": int(cls),
                                 "box_caption": "%s %.3f" % (names[cls], conf),
                                 "scores": {"class_score": conf},
                                 "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                    boxes = {"predictions": {"box_data": box_data, "class_labels": names}}
                    wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Save predictions to COCO JSON format
            if save_json:
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # Convert to xywh format
                box[:, :2] -= box[:, 2:] / 2  # Convert center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign predictions as incorrect initially
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:  # If ground truth exists
                detected = []  # Detected target indices
                tcls_tensor = labels[:, 0]  # Ground truth classes

                # Scale ground truth boxes to image space
                tbox = xywh2xyxy(labels[:, 1:5])  # Convert to xyxy format
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # Scale to native-space labels

                # Process confusion matrix if enabled
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Process predictions for each class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # Ground truth indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # Prediction indices

                    # Match predictions to targets
                    if pi.shape[0]:  # If predictions exist
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # Compute IoUs and get best match
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):  # Check IoUs above threshold
                            d = ti[i[j]]  # Target index
                            if d.item() not in detected_set:  # Ensure unique detection
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # Mark prediction as correct
                                if len(detected) == nl:  # All targets detected
                                    break

            # Append statistics for this image
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Generate and save visualizations
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # Labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # Predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
