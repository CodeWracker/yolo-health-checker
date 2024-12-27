# yolo-health-checker

A Python tool to perform an in-depth analysis of YOLO-format datasets, providing both **multidimensional** and **unidimensional** metrics of dataset “health.” This package helps you **quantify** class distribution, spatial distribution, and other properties in a systematic and reproducible way.

## Features

- **Class Distribution Analysis**:  
  - Calculate entropy, Gini index, and standard deviation of instance counts per class.
  - Inspect the number of instances per class and identify imbalances quickly.

- **Spatial Distribution Analysis**:  
  - Compute spatial entropy of bounding boxes to see how they are spread across images.
  - Measure standard deviation of bounding box centers, detecting if objects cluster in certain image regions.
  - Calculate average distance from image center, unveiling potential “center bias” in your dataset.

- **Rich Visual Outputs**:  
  - Automatically generates heatmaps of bounding box footprints and centers.
  - Visual bar charts showing the number of instances per class.

- **Comprehensive Logging**:  
  - Each analysis is logged into a log file, capturing potential warnings (e.g., missing annotations) and key statistics.

- **Modular**:  
  - Integrate directly into Python code, or run as a command-line script.
  - Produces CSV reports of class distribution and overall health metrics.

## Installation

Install `yolo-health-checker` via pip:

``pip install yolo-health-checker``

## Usage

There are two main ways to use this tool: **as a command-line script** or **via Python import**.

### Command-Line

``python -m yolo_health_checker.analyze_dataset /path/to/yolo_dataset --output_dir results --save_images --save_csv``

- `dataset_path`: The path to your YOLO-format dataset (containing `data.yaml`, `train/`, `val/` folders).
- `--output_dir`: Optional path to store the output artifacts (CSV, images, etc.).
- `--save_images`: Save class distribution bar charts and heatmaps.
- `--save_csv`: Save CSV files with class distributions and health metrics.
- `--log_file`: Specify the log file name (default: `main.log`).
- `--log_level`: Logging verbosity (default: `INFO`).

### Python Import

You can also integrate `yolo-health-checker` within your Python code:

```
from yolo_health_checker import analyze_dataset

health_checker = analyze_dataset(
    dataset_path='/path/to/yolo_dataset',
    output_dir='results',
    save_images=True,
    save_csv=True,
    log_level='INFO',
    log_file='analysis.log'
)

# Once analysis is done, inspect the results
health_checker.show_health_metrics()
```

## Motivations & Numeric Measurements

> **Why numeric measurements?**  
> Numeric metrics allow us to systematically compare how well different YOLO versions handle dataset variations. By converting each characteristic into a **measurable number**, we make the research both **reproducible** and **statistically testable**.

Below we list the main “dataset health” metrics we measure. Each is **numeric** with a clear interpretation, making them suitable for statistical analyses. The overarching principle: **if it cannot be expressed numerically, we cannot reliably correlate it with YOLO performance**.

### 1. Class Distribution Metrics

#### 1.1. Entropy of Class Distribution
- **Reason:** Measures the **uniformity** of the distribution of objects across classes. A high entropy indicates a more balanced dataset.
- **Formula:**  
  H = - Σ pᵢ log(pᵢ)  
  (where pᵢ is the proportion of class i)

#### 1.2. Gini Index
- **Reason:** Captures how **unevenly** instances are distributed among classes.
- **Formula:**  
  G = 1 - Σ (pᵢ)²  
  (where pᵢ is the proportion of class i)

#### 1.3. Standard Deviation of Instances per Class
- **Reason:** Indicates the spread of counts across different classes. 

### 2. Spatial Distribution Metrics

#### 2.1. Entropy of Object Locations
- **Reason:** Checks if bounding boxes are **clustered** in a few regions or **spread** evenly.
- **Procedure:** A 10×10 grid is created, and bounding box counts per cell are transformed into probabilities for entropy calculation.

#### 2.2. Standard Deviation of Object Centers
- **Reason:** Measures how widely scattered the center points of bounding boxes are across the image.

#### 2.3. Distance from Center of Mass
- **Reason:** Quantifies how far bounding box centers lie from the image center, highlighting potential “center bias.”

## Example

To run a sample analysis (outputting both CSV and images):

``python -m yolo_health_checker.analyze_dataset /path/to/yolo_dataset --output_dir results --save_images --save_csv --log_file my_log.log``

- This will log the process in **my_log.log**, generate bar charts for class distribution, produce bounding box heatmaps, and create CSV reports with class counts and dataset health metrics in the `results/health` folder.

## Contributing

Feel free to open an issue or a pull request if you spot bugs or want to contribute improvements. We welcome new ideas on metrics or enhancements to support more YOLO-format variations.

## License

This project is licensed under an open-source license (e.g., MIT, Apache 2.0, etc.). Check the repository for more details.

Happy analyzing with **yolo-health-checker**!
