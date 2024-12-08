# YOLOv7 Setup and Usage Guide

This guide provides step-by-step instructions for setting up and running our modified YOLOv7 on your system, focusing on the following tasks:
1. Downloading the code onto the system.
2. Setting up the environment.
3. Setting up the dataset.
4. Running `test.py` with pretrained weights.
5. Running a training run and optionally creating a smaller dataset for quick experiments.
6. Viewing the results.

---

## Step 1: Downloading the Code
1. Clone the YOLOv7 repository:
   ```bash
   git clone https://github.com/AyoRIT/YOLOV7.git
   cd YOLOV7
   ```

2. Ensure that all files have been downloaded, including the `test.py`, `train.py`, and configuration files.

---

## Step 2: Setting Up the Environment
1. Install Python 3.8 or higher. (This process will require a GPU)
2. Set up a virtual environment:
   ```bash
   python3 -m venv yolov7_env
   source yolov7_env/bin/activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 3: Setting Up the Dataset
1. **COCO-mini Dataset Setup**:
   - For ease of use, a mini subset of COCO has already been prepared for your convenience. This subset includes approximately 500 training images and 50 validation images, along with a small set of test images.
   - IMPORTANT: Using this heavily reduced dataset will not produce replicated results from our presentation. This dataset is used just as a portable proof of concept in terms of being able top train.
   - Ensure that the following directory structure exists under `coco_mini/`:

     ```
     ./coco_mini/
     ├── images/
     │   ├── train2017_subset/  # Contains ~500 training images
     │   ├── val2017_subset/    # Contains ~50 validation images
     │   └── test2017_subset/   # Contains a small set of test images
     ├── labels/
     │   ├── train2017_subset/
     │   └── val2017_subset/
     └── annotations/
     ```

   - If the `coco_mini/` directory does not exist or is incomplete, ensure the files are set up properly before proceeding. The dataset is pre-configured to work seamlessly with `coco_mini/`.

---

## Step 4: Running `test.py` with Pretrained Weights
1. A checkpoint containing the pretrained weights for the original YOLOv7 model has already been provided under `checkpoints/`. Ensure that the file `yolov7.pt` exists in this directory. If the file is missing, download it manually from the official YOLOv7 repository.

   Ensure the following directory structure:
   ```
   checkpoints/
   └── yolov7.pt
   ```

2. Run `test.py` to evaluate the model using the provided pretrained weights:
   ```bash
   python test.py --data data/coco.yaml --img 640 --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights checkpoints/yolov7.pt --name yolov7_eval
   ```

---

## Step 5: Running a Training Run on Original Model From Scratch
1. Start a training run for 1 epoch:
   ```bash
   python train.py --workers 8 --device 0 --batch-size 8 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7_train --hyp data/hyp.scratch.p5.yaml --epochs 1
   ```

2. **Optional: Using the Modified Architecture**:
   - Start a training run for 1 epoch with the additional convolutional layer within the backbone architecture:
    ```bash
    python train.py --workers 8 --device 0 --batch-size 8 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7_ML.yaml --weights '' --name yolov7_modified_train --hyp data/hyp.scratch.p5.yaml --epochs 1
     ```

---

## Step 6: Viewing Results
1. Training results, including loss metrics and performance graphs, are saved in the `runs/train/{experiment_name}` directory.
   - Example:
     ```
     runs/train/yolov7_train/
     ├── events.out.tfevents...  # TensorBoard logs
     ├── results.png             # Training results graph
     ├── weights/
     │   ├── last.pt             # Latest model weights
     │   ├── best.pt             # Best-performing model weights
     ```

2. To visualize training progress:
   - Use TensorBoard:
     ```bash
     tensorboard --bind_all --logdir runs/train/<experiment_name> --port 6006
     ```

3. Testing results are saved in the `runs/test/{experiment_name}` directory:
   - This directory co
   - Example:
     ```
     runs/test/yolov7_eval/
     ├── results.txt             # Evaluation metrics
     ├── confusion_matrix.png    # Confusion matrix visualization
     ├── PR_curve.png            # Precision-recall curve
     ```

---

## Notes and Tips
- **Hardware Requirements**:
   - For full COCO training, a GPU is required.

- **Configuring Training Parameters**:
   - Edit `data/hyp.scratch.p5.yaml` to adjust hyperparameters such as learning rate and augmentation settings.

- **Common Errors**:
   - Ensure all dataset paths in `data/coco.yaml` exist and contain the correct data.
   - Verify the environment dependencies match those listed in `requirements.txt`.

--- 