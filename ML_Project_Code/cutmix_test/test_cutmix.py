import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def load_image_and_labels(image_path, label_path):
    """
    Load an image and its corresponding labels from the given paths.
    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.

    Returns:
        image (ndarray): Loaded image.
        labels (list): Parsed labels (class, x_center, y_center, width, height or polygons).
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            labels.append(list(map(float, line.strip().split())))
    return image, labels  # Returning `labels` as a list of lists for flexibility


def plot_image_with_labels(image, labels, title="Image with Labels"):
    """
    Plot an image with bounding box or polygon labels.
    Args:
        image (ndarray): Image to plot.
        labels (list): Bounding box or polygon labels.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    h, w, _ = image.shape
    for label in labels:
        cls, *coords = label
        coords = np.array(coords).reshape(-1, 2)  # Reshape to pairs of (x, y)
        coords[:, 0] *= w  # Scale x to image width
        coords[:, 1] *= h  # Scale y to image height
        plt.plot(*coords.T, marker="o", linestyle="-", color="red", linewidth=2)
        plt.text(
            coords[0, 0],
            coords[0, 1] - 10,
            f"Class {int(cls)}",
            color="yellow",
            fontsize=12,
            bbox=dict(facecolor="blue", alpha=0.5),
        )
    plt.title(title)
    plt.axis("off")
    plt.show()


import cv2

def cutmix(image, label, second_image, second_label, beta=1.0):
    """
    Apply CutMix augmentation with proper label adjustment for polygon labels.
    Args:
        image (ndarray): The first image.
        label (list): Labels corresponding to the first image (polygons).
        second_image (ndarray): The second image to mix with.
        second_label (list): Labels corresponding to the second image (polygons).
        beta (float): The Beta distribution parameter for mixing ratio.

    Returns:
        mixed_image (ndarray): The augmented image with parts from both images.
        mixed_label (list): Adjusted labels for the mixed image.
    """
    # Resize second_image to match image dimensions
    h, w = image.shape[:2]
    second_image = cv2.resize(second_image, (w, h))  # Resize to (width, height)

    lam = np.random.beta(beta, beta)

    # Random center point for the CutMix region
    cx, cy = np.random.randint(w), np.random.randint(h)

    # Calculate the CutMix bounding box
    bbx1 = np.clip(cx - int(w * np.sqrt(1 - lam)), 0, w)
    bby1 = np.clip(cy - int(h * np.sqrt(1 - lam)), 0, h)
    bbx2 = np.clip(cx + int(w * np.sqrt(1 - lam)), 0, w)
    bby2 = np.clip(cy + int(h * np.sqrt(1 - lam)), 0, h)

    # Mix the images
    mixed_image = image.copy()
    mixed_image[bby1:bby2, bbx1:bbx2, :] = second_image[bby1:bby2, bbx1:bbx2, :]

    # Adjust labels for both images
    mixed_label = []

    # Retain labels from the first image
    for obj in label:
        class_id = obj[0]
        polygon = np.array(obj[1:]).reshape(-1, 2)

        # Compute bounding box
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)

        # Check if the bounding box is outside the CutMix region
        if x_max < bbx1 / w or x_min > bbx2 / w or y_max < bby1 / h or y_min > bby2 / h:
            mixed_label.append(obj)

    # Adjust and retain labels from the second image
    for obj in second_label:
        class_id = obj[0]
        polygon = np.array(obj[1:]).reshape(-1, 2)

        # Compute bounding box
        x_min, y_min = polygon.min(axis=0)
        x_max, y_max = polygon.max(axis=0)

        # Clip bounding box coordinates to the CutMix region
        x_min = max(x_min, bbx1 / w)
        x_max = min(x_max, bbx2 / w)
        y_min = max(y_min, bby1 / h)
        y_max = min(y_max, bby2 / h)

        if x_min < x_max and y_min < y_max:  # If the bounding box is valid
            new_polygon = polygon.copy()

            # Clip polygon coordinates to the CutMix region
            new_polygon[:, 0] = np.clip(new_polygon[:, 0], bbx1 / w, bbx2 / w)
            new_polygon[:, 1] = np.clip(new_polygon[:, 1], bby1 / h, bby2 / h)

            # Flatten the polygon and append the class ID
            flattened_polygon = new_polygon.flatten().tolist()
            mixed_label.append([class_id] + flattened_polygon)

    return mixed_image, mixed_label



# Paths
image_dir = "./images/"
label_dir = "./labels/"

# Load image and label paths
image_files = sorted(
    [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
)
label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".txt")])

# Sanity check: Ensure images and labels correspond
assert len(image_files) == len(label_files), "Mismatch between image and label counts."

# Select two random images and labels
idx1 = random.randint(0, len(image_files) - 1)
idx2 = random.randint(0, len(image_files) - 1)
image1, labels1 = load_image_and_labels(
    os.path.join(image_dir, image_files[idx1]),
    os.path.join(label_dir, label_files[idx1]),
)
image2, labels2 = load_image_and_labels(
    os.path.join(image_dir, image_files[idx2]),
    os.path.join(label_dir, label_files[idx2]),
)

# Show original images with labels
plot_image_with_labels(image1, labels1, title="Original Image 1")
plot_image_with_labels(image2, labels2, title="Original Image 2")

# Apply CutMix
mixed_image, mixed_labels = cutmix(image1, labels1, image2, labels2, beta=1.0)

# Show augmented image with labels
plot_image_with_labels(mixed_image, mixed_labels, title="Augmented Image with CutMix")
