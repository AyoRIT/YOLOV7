import json
import os
import random
from shutil import copyfile

def create_coco_subset(original_json, output_json, image_dir, output_dir, num_images):
    # Load original COCO annotations
    with open(original_json, 'r') as f:
        coco_data = json.load(f)

    # Select a random subset of images
    selected_images = random.sample(coco_data['images'], num_images)
    selected_image_ids = {img['id'] for img in selected_images}

    # Filter annotations for selected images
    selected_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in selected_image_ids]

    # Create a new COCO JSON
    coco_subset = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco_data['categories']
    }

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_json), exist_ok=True)  # Create parent dirs for JSON
    os.makedirs(output_dir, exist_ok=True)  # Create image output dir

    # Save the new JSON
    with open(output_json, 'w') as f:
        json.dump(coco_subset, f, indent=4)

    # Copy selected images to the output directory
    for img in selected_images:
        src_path = os.path.join(image_dir, img['file_name'])
        dest_path = os.path.join(output_dir, img['file_name'])
        copyfile(src_path, dest_path)

    print(f"Subset created with {num_images} images. JSON saved to {output_json}.")


# Paths
train_original_json = "../../coco/annotations/annotations/instances_train2017.json"
val_original_json = "../../coco/annotations/annotations/instances_val2017.json"

train_image_dir = "../../coco/images/train2017"
val_image_dir = "../../coco/images/val2017"

# Output directories
base_output_dir = "../../coco_mini"
train_output_json = os.path.join(base_output_dir, "annotations", "instances_train2017_subset.json")
val_output_json = os.path.join(base_output_dir, "annotations", "instances_val2017_subset.json")
train_output_dir = os.path.join(base_output_dir, "images", "train2017_subset")
val_output_dir = os.path.join(base_output_dir, "images", "val2017_subset")

# Number of images for each subset
train_images = 500  # Adjust as needed
val_images = 50

# Create subsets
create_coco_subset(train_original_json, train_output_json, train_image_dir, train_output_dir, train_images)
create_coco_subset(val_original_json, val_output_json, val_image_dir, val_output_dir, val_images)
