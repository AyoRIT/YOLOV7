import json
import os
import random
from shutil import copyfile


def create_coco_subset_from_labels(original_json, output_json, image_dir, label_dir, output_image_dir, output_label_dir, num_images):
    # Load original COCO annotations
    with open(original_json, 'r') as f:
        coco_data = json.load(f)

    # Gather all label files in the labels directory
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    label_file_names = {os.path.splitext(f)[0] for f in label_files}  # Extract base names of label files

    # Filter images to only include those with corresponding label files
    valid_images = [img for img in coco_data['images'] if os.path.splitext(img['file_name'])[0] in label_file_names]

    # Warn if no valid images are found
    if not valid_images:
        print(f"Warning: No valid images found with matching labels in {label_dir}.")
        return

    # Select a random subset of valid images
    selected_images = random.sample(valid_images, min(num_images, len(valid_images)))
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
    os.makedirs(output_image_dir, exist_ok=True)  # Create output image dir
    os.makedirs(output_label_dir, exist_ok=True)  # Create output label dir

    # Save the new JSON
    with open(output_json, 'w') as f:
        json.dump(coco_subset, f, indent=4)

    # Copy selected images and corresponding labels
    for img in selected_images:
        # Copy image
        src_image_path = os.path.join(image_dir, img['file_name'])
        dest_image_path = os.path.join(output_image_dir, img['file_name'])
        copyfile(src_image_path, dest_image_path)

        # Copy corresponding label
        label_file_name = os.path.splitext(img['file_name'])[0] + ".txt"
        src_label_path = os.path.join(label_dir, label_file_name)
        dest_label_path = os.path.join(output_label_dir, label_file_name)

        copyfile(src_label_path, dest_label_path)

    print(f"Subset created with {len(selected_images)} images and their labels. JSON saved to {output_json}.")


# Paths for training, validation
base_input_dir = "../../coco"
base_output_dir = "../../coco_mini"

# Train
train_original_json = os.path.join(base_input_dir, "annotations/annotations/instances_train2017.json")
train_image_dir = os.path.join(base_input_dir, "images/train2017")
train_label_dir = os.path.join(base_input_dir, "labels/train2017")
train_output_json = os.path.join(base_output_dir, "annotations/instances_train2017_subset.json")
train_output_image_dir = os.path.join(base_output_dir, "images/train2017_subset")
train_output_label_dir = os.path.join(base_output_dir, "labels/train2017_subset")

# Validation
val_original_json = os.path.join(base_input_dir, "annotations/annotations/instances_val2017.json")
val_image_dir = os.path.join(base_input_dir, "images/val2017")
val_label_dir = os.path.join(base_input_dir, "labels/val2017")
val_output_json = os.path.join(base_output_dir, "annotations/instances_val2017_subset.json")
val_output_image_dir = os.path.join(base_output_dir, "images/val2017_subset")
val_output_label_dir = os.path.join(base_output_dir, "labels/val2017_subset")

# Number of images for each subset
train_images = 500  # Adjust as needed
val_images = 50

# Create subsets
create_coco_subset_from_labels(train_original_json, train_output_json, train_image_dir, train_label_dir,
                               train_output_image_dir, train_output_label_dir, train_images)
create_coco_subset_from_labels(val_original_json, val_output_json, val_image_dir, val_label_dir,
                               val_output_image_dir, val_output_label_dir, val_images)
