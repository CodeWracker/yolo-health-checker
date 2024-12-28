"""
YOLO Dataset Health Analysis

This script examines a YOLO-format dataset and outputs both **Health Metrics** 
(multidimensional data) and **Health Parameters** (unidimensional data), 
enabling a comprehensive “health check” of the dataset. 

--------------------------------------------------------------------
Health Metrics (Multidimensional)
--------------------------------------------------------------------
- **Class Distribution** (number of instances per class): saved as CSV
- **Heatmaps**:
  1. **Bounding Box Heatmap** (shows the spatial footprint of the entire box)
  2. **Bounding Box Centers Heatmap** (shows only the centers of boxes)
- **Bounding Box Centers** (raw list in memory, if needed for extended analysis)

--------------------------------------------------------------------
Health Parameters (Unidimensional)
--------------------------------------------------------------------
These are compiled into `health_metrics.csv`. Examples include:
- **Number of Classes** 
- **Total Images**
- **Total Annotations**
- **Number of Images Without Annotations**
- **Number of Empty (Null) Annotations**
- **Class Distribution Metrics**:
  - Gini Index
  - Shannon Entropy
  - Standard Deviation of Class Counts
- **Spatial Distribution Metrics**:
  - Spatial Entropy of Object Locations
  - Standard Deviation of Object Centers
  - Distance from Center of Mass
- Other single-value indicators relevant to dataset health

--------------------------------------------------------------------
Additional Outputs
--------------------------------------------------------------------
- Bar chart plots for class distribution
- Heatmaps for bounding boxes and bounding box centers
- Logging information for each processing step (stored in `main.log`)

--------------------------------------------------------------------
Usage Example:
    python analyze_dataset.py /path/to/yolo_dataset
--------------------------------------------------------------------
"""

import os
import yaml as pyyaml
import csv
import math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm
import pandas as pd

# -------------------------------------------------------
# Class Definition
# -------------------------------------------------------
class HealthChecker:
    """
    A class to store and expose the results of the YOLO dataset analysis.
    """
    def __init__(self):
        self.data = {
            'train': {
            'class_distribution': pd.DataFrame(),
            'health_metrics': pd.DataFrame(),
            'bbox_heatmap': None,
            'center_heatmap': None
        },
            'val': {
                'class_distribution': pd.DataFrame(),
                'health_metrics': pd.DataFrame(),
                'bbox_heatmap': None,
                'center_heatmap': None
            }
        }
        
        
    def show_health_metrics(self):
        """
        Display the health metrics for each split.
        """
        print('Train Split Metrics:')
        print(self.data['train']['health_metrics'].head())
        print(self.data['train']['class_distribution'].head())
        print('\nVal Split Metrics:')
        print(self.data['val']['health_metrics'].head())
        print(self.data['val']['class_distribution'].head())
        

# -------------------------------------------------------
# 1. Class Distribution Metrics
# -------------------------------------------------------

def compute_gini_index(class_counts: dict) -> float:
    """
    Computes the Gini Index for the class distribution.
    Gini Index ranges from 0 (extreme imbalance) to 1 (perfect balance).
    """
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    proportions = [count / total for count in class_counts.values()]
    gini = 1 - sum([p**2 for p in proportions])
    return round(gini, 6)

def compute_entropy_class_distribution(class_counts: dict) -> float:
    """
    Computes Shannon Entropy for class distribution.
    High entropy indicates more uniform class distribution.
    """
    total = sum(class_counts.values())
    if total == 0:
        return 0.0
    proportions = [count / total for count in class_counts.values()]
    entropy = -sum([p * math.log(p + 1e-9) for p in proportions])
    return round(entropy, 6)

def compute_std_class_counts(class_counts: dict) -> float:
    """
    Computes the standard deviation of the per-class instance counts.
    """
    counts = list(class_counts.values())
    if len(counts) == 0:
        return 0.0
    return round(float(np.std(counts)), 6)

def compute_num_classes(class_counts: dict) -> int:
    """
    Returns the number of classes that have at least one instance (count > 0).
    """
    return sum(1 for c in class_counts.values() if c > 0)

# -------------------------------------------------------
# 2. Spatial Distribution Metrics
# -------------------------------------------------------

def compute_spatial_entropy(bboxes_centers: list, grid_size=10) -> float:
    """
    Computes the entropy of object locations by dividing the image into
    a grid of size `grid_size x grid_size`. 

    Parameters:
        bboxes_centers (list): List of (x_center, y_center) in normalized [0,1].
        grid_size (int): Number of grid cells per dimension.

    Returns:
        float: Spatial entropy. High if objects are spread out, low if concentrated.
    """
    if not bboxes_centers:
        return 0.0

    # Create a grid of zeros (grid_size x grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for (xc, yc) in bboxes_centers:
        gx = int(xc * grid_size)
        gy = int(yc * grid_size)
        gx = min(gx, grid_size - 1)
        gy = min(gy, grid_size - 1)
        grid[gy, gx] += 1.0

    total_boxes = len(bboxes_centers)
    grid_probs = grid.flatten() / (total_boxes + 1e-9)

    entropy = -sum([p * math.log(p + 1e-9) for p in grid_probs])
    return round(entropy, 6)

def compute_std_object_centers(bboxes_centers: list) -> float:
    """
    Computes the combined standard deviation of object center coordinates.

    D = sqrt(std_x^2 + std_y^2)

    If centers are clustered, this value will be low.
    If they are spread out, it will be high.
    """
    if not bboxes_centers:
        return 0.0

    xs = [b[0] for b in bboxes_centers]
    ys = [b[1] for b in bboxes_centers]

    std_x = float(np.std(xs))
    std_y = float(np.std(ys))
    D = math.sqrt(std_x**2 + std_y**2)
    return round(D, 6)

def compute_distance_from_center_of_mass(bboxes_centers: list) -> float:
    """
    Computes the average distance of object centers from the image center (0.5, 0.5).
    Alternatively, we could compute the distance from the 'center of mass' of 
    all objects, but here we consider the geometric image center by default.

    D_cm = mean( sqrt( (x_i - 0.5)^2 + (y_i - 0.5)^2 ) )
    """
    if not bboxes_centers:
        return 0.0

    distances = []
    for (xc, yc) in bboxes_centers:
        dist = math.sqrt((xc - 0.5)**2 + (yc - 0.5)**2)
        distances.append(dist)

    return round(float(np.mean(distances)), 6)



# -------------------------------------------------------
# Main Analysis Function
# -------------------------------------------------------

def analyze_dataset(
    dataset_path: str, 
    output_dir=None, 
    save_images=False, 
    save_csv=False, 
    log_level='INFO', 
    log_file='main.log'
):
    """
    Main function that analyzes a YOLO-format dataset at `dataset_path`.
    - Generates raw CSV files with class distribution, etc.
    - Computes and saves unidimensional metrics in `health_metrics.csv`.
    - Generates plots and logs for manual inspection:
      * Class distribution bar charts
      * Full bounding-box heatmap (1000x1000)
      * Bounding-box center heatmap (1000x1000)

    Parameters:
    -------------
    dataset_path: str
        Path to the YOLO dataset.
    output_dir: str, optional
        Directory to save outputs. If None, no files are saved.
    save_images: bool, optional
        If True, saves the plots (class distribution, heatmaps).
    save_csv: bool, optional
        If True, saves CSV reports.
    log_level: str, optional
        Logging level.
    log_file: str, optional
        Log file path.
    """
    health_checker = HealthChecker()

    # Configure logging here so it uses the function's parameters
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )

    if(output_dir is not None and (not save_images and not save_csv)):
        logging.warning('Output directory is provided but no outputs are enabled. Exiting.')
        return health_checker

    dataset_path = dataset_path.replace('\\', '/')
    logging.info(f'Starting dataset analysis at: {dataset_path}')

    # Load data.yaml
    data_yaml_path = os.path.join(dataset_path, 'data.yaml').replace('\\', '/')
    try:
        with open(data_yaml_path, 'r') as f_yaml:
            yaml_data = pyyaml.safe_load(f_yaml)
        logging.info('Loaded data.yaml successfully.')
    except Exception as e:
        logging.error(f'Error loading data.yaml: {e}')
        raise

    class_names = yaml_data.get('names', [])
    # We assume only train/val splits here, but can be extended if needed
    splits = {
        'train': os.path.join(dataset_path, 'train'),
        'val': os.path.join(dataset_path, 'val')
    }

    # Dictionary to store results for each split
    results = {}

    # If the user passed an output_dir, create the "health" folder inside it.
    health_folder = None
    if output_dir is not None:
        health_folder = os.path.join(output_dir, 'health')
        os.makedirs(health_folder, exist_ok=True)

    for split_name, split_path in splits.items():
        logging.info(f'Analyzing split: {split_name}')
        images_path = os.path.join(split_path, 'images')
        labels_path = os.path.join(split_path, 'labels')
        
        # check if the images and labels folders exist, if not, continue to the next split
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            logging.warning(f'Missing images or labels folder for split: {split_name}')
            continue

        total_images = 0
        total_annotations = 0
        images_without_annotation = 0
        empty_annotations = 0
        class_counter = Counter()

        # We'll store bounding box centers (x, y) in normalized coords
        bbox_centers = []

        # 1000x1000 heatmap for the entire bounding box footprint
        heatmap_bboxes = np.zeros((1000, 1000), dtype=np.float32)
        # 1000x1000 heatmap specifically for bounding box centers
        heatmap_centers = np.zeros((1000, 1000), dtype=np.float32)

        image_files = [
            f for f in os.listdir(images_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        for img_file in tqdm(image_files, desc=f'Processing {split_name}', unit='img'):
            total_images += 1
            label_file = os.path.join(
                labels_path, os.path.splitext(img_file)[0] + '.txt'
            )

            if not os.path.exists(label_file):
                images_without_annotation += 1
                logging.warning(f'Missing annotation for image: {img_file}')
                continue

            try:
                with open(label_file, 'r') as lf:
                    lines = lf.readlines()
                    if not lines:
                        empty_annotations += 1
                        logging.info(f'Empty annotation for image: {img_file}')
                        continue

                    for line in lines:
                        total_annotations += 1
                        data = line.strip().split()
                        class_id = int(data[0])
                        class_counter[class_id] += 1

                        x_center = float(data[1])
                        y_center = float(data[2])
                        w = float(data[3])
                        h = float(data[4])

                        # Store normalized center for spatial metrics
                        bbox_centers.append((x_center, y_center))

                        # Convert center to integer for the center heatmap
                        cx = int(x_center * 1000)
                        cy = int(y_center * 1000)
                        if 0 <= cx < 1000 and 0 <= cy < 1000:
                            heatmap_centers[cy, cx] += 1.0

                        # Convert bounding box footprint to integer coords
                        x_min = int((x_center - w / 2) * 1000)
                        x_max = int((x_center + w / 2) * 1000)
                        y_min = int((y_center - h / 2) * 1000)
                        y_max = int((y_center + h / 2) * 1000)

                        # Clamp to avoid out-of-bounds
                        x_min = max(x_min, 0)
                        x_max = min(x_max, 999)
                        y_min = max(y_min, 0)
                        y_max = min(y_max, 999)

                        # Increment bounding box footprint
                        heatmap_bboxes[y_min:y_max+1, x_min:x_max+1] += 1.0

            except Exception as e:
                logging.warning(f'Error processing file {img_file}: {e}')

        # Prepare result dictionary for this split
        results[split_name] = {
            'total_images': total_images,
            'total_annotations': total_annotations,
            'images_without_annotation': images_without_annotation,
            'empty_annotations': empty_annotations,
            'class_counts': dict(class_counter),
            'bbox_centers': bbox_centers  # used to compute spatial metrics
        }
        
        health_checker.data[split_name]['class_distribution'] = pd.DataFrame({
            'class_id': sorted(class_counter.keys()),
            'count': [class_counter[cid] for cid in sorted(class_counter.keys())]
        })
        health_checker.data[split_name]['bbox_heatmap'] = heatmap_bboxes
        health_checker.data[split_name]['center_heatmap'] = heatmap_centers

        logging.info(f'{split_name} - Images: {total_images}')
        logging.info(f'{split_name} - Annotations: {total_annotations}')
        logging.info(f'{split_name} - Images w/o annotation: {images_without_annotation}')
        logging.info(f'{split_name} - Empty annotations: {empty_annotations}')

        # -------------------------------------------------------
        # Save class distribution if output_dir is not None and save_csv is True
        # -------------------------------------------------------
        if health_folder is not None and save_csv:
            csv_class_dist = os.path.join(health_folder, f'class_distribution_{split_name}.csv')
            df_class_dist = health_checker.data[split_name]['class_distribution']
            df_class_dist.to_csv(csv_class_dist, index=False, encoding='utf-8')
            logging.info(f'Saved class distribution CSV for {split_name} -> {csv_class_dist}')

        # -------------------------------------------------------
        # Plot and save class distribution as PNG if output_dir is not None and save_images is True
        # -------------------------------------------------------
        class_ids = sorted(class_counter.keys())
        class_counts = [class_counter[cid] for cid in class_ids]
        if health_folder is not None and save_images:
            plt.figure(figsize=(8, 6))
            plt.bar(class_ids, class_counts, color='blue')
            plt.xlabel('Class ID')
            plt.ylabel('Count')
            plt.title(f'Class Distribution - {split_name}')
            plt.xticks(class_ids)  # might be cluttered if many classes
            dist_plot_file = os.path.join(health_folder, f'class_distribution_{split_name}.png')
            plt.savefig(dist_plot_file)
            plt.close()
            logging.info(f'Saved class distribution plot for {split_name} -> {dist_plot_file}')

        # -------------------------------------------------------
        # Save bounding-box footprint heatmap as PNG if output_dir is not None and save_images is True
        # -------------------------------------------------------
        if health_folder is not None and save_images:
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmap_bboxes, cmap='hot', interpolation='nearest', origin='lower')
            plt.title(f'Annotation BBox Heatmap - {split_name}')
            plt.colorbar(label='Number of bounding boxes')
            heatmap_file = os.path.join(health_folder, f'heatmap_bboxes_{split_name}.png')
            plt.savefig(heatmap_file)
            plt.close()
            logging.info(f'Saved bounding-box footprint heatmap for {split_name} -> {heatmap_file}')

        # -------------------------------------------------------
        # Save bounding-box center heatmap as PNG if output_dir is not None and save_images is True
        # -------------------------------------------------------
        if health_folder is not None and save_images:
            plt.figure(figsize=(6, 6))
            plt.imshow(heatmap_centers, cmap='hot', interpolation='nearest', origin='lower')
            plt.title(f'Annotation Centers Heatmap - {split_name}')
            plt.colorbar(label='Number of bounding box centers')
            center_heatmap_file = os.path.join(health_folder, f'heatmap_centers_{split_name}.png')
            plt.savefig(center_heatmap_file)
            plt.close()
            logging.info(f'Saved bounding-box center heatmap for {split_name} -> {center_heatmap_file}')

    # -------------------------------------------------------
    # Compute unidimensional metrics and save in health_metrics.csv (if output_dir is not None and save_csv is True)
    # -------------------------------------------------------
    rows = []
    for split_name, info in results.items():
        c_counts = info['class_counts']
        centers = info['bbox_centers']

        # Class distribution metrics
        effective_num_classes = compute_num_classes(c_counts)
        gini_index = compute_gini_index(c_counts)
        entropy_class_dist = compute_entropy_class_distribution(c_counts)
        std_class = compute_std_class_counts(c_counts)

        # Spatial metrics
        spatial_entropy_val = compute_spatial_entropy(centers, grid_size=10)
        std_centers = compute_std_object_centers(centers)
        distance_cm = compute_distance_from_center_of_mass(centers)

        rows.append([
            split_name,
            info['total_images'],
            info['total_annotations'],
            info['images_without_annotation'],
            info['empty_annotations'],
            effective_num_classes,
            gini_index,
            entropy_class_dist,
            std_class,
            spatial_entropy_val,
            std_centers,
            distance_cm
        ])

    df_metrics = pd.DataFrame(rows, columns=[
            'split',
            'total_images',
            'total_annotations',
            'images_without_annotation',
            'empty_annotations',
            'effective_num_classes',
            'gini_index',
            'entropy_class_dist',
            'std_class_counts',
            'spatial_entropy',
            'std_object_centers',
            'avg_distance_center_of_mass'
        ])
    if health_folder is not None and save_csv:
        metrics_csv_path = os.path.join(health_folder, 'health_metrics.csv')
        df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8')
        
        logging.info(f'Saved unidimensional metrics in {metrics_csv_path}')

    logging.info('Analysis completed successfully.')

    # save the metrics inside the health_checker object
    health_checker.health_metrics = df_metrics
    return health_checker


# -------------------------------------------------------
# Entry Point
# -------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Dataset Health Analysis')
    parser.add_argument('dataset_path', type=str, help='Path to the YOLO dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--save_images', action='store_true', help='Save images')
    parser.add_argument('--save_csv', action='store_true', help='Save CSV reports')
    parser.add_argument('--log_level', type=str, default='INFO', help='Log level')
    parser.add_argument('--log_file', type=str, default='main.log', help='Path to the log file')

    args = parser.parse_args()

    health_checker = analyze_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        save_images=args.save_images,
        save_csv=args.save_csv,
        log_level=args.log_level,
        log_file=args.log_file
    )

    health_checker.show_health_metrics()