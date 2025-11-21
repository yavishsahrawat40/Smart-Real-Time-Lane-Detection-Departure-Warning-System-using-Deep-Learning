# An Efficient Deep Learning Framework for Real-Time Lane Segmentation and Departure Warning

A comprehensive deep learning-based perception system combining lane detection, lane departure warning (LDW), and adjacent lane obstacle detection using U-Net and Attention U-Net architectures, trained on CULane and TuSimple datasets for Advanced Driver Assistance Systems (ADAS) and autonomous vehicles.

**Authors**: Yavish Sahrawat, Janhvi Singh, Shubham Gangwar, G N S S Abhiram  
**Guide**: Dr. Tauseef Khan

## Overview

This project implements an integrated perception pipeline that goes beyond simple lane detection. It combines semantic segmentation for lane detection with practical safety features including lane departure warnings and adjacent lane obstacle detection using YOLOv8s. The system addresses real-world challenges like varying lighting conditions, weather changes, and traffic scenarios through established deep learning architectures.

## Key Features

### 🚗 Integrated Perception System
- **Lane Detection**: Semantic segmentation using U-Net and Attention U-Net architectures
- **Lane Departure Warning (LDW)**: Real-time alerts when vehicle drifts beyond 35% of lane width
- **Adjacent Lane Obstacle Detection**: YOLOv8s-based vehicle detection with lane-aware positioning

### 🧠 Model Architectures
- **Baseline U-Net**: Classic encoder-decoder with skip connections for spatial detail preservation
- **Attention U-Net**: Enhanced with attention gates to focus on lane markings while filtering out distractions (shadows, trees, buildings)
- Trained on both TuSimple (highway) and CULane (challenging urban/weather conditions)

### 🔧 Robust Training Pipeline
- Mixed precision training (AMP) for faster computation
- Advanced data augmentation using Albumentations library
- Combined BCE + Dice loss for handling class imbalance
- ReduceLROnPlateau scheduler for adaptive learning
- Checkpoint saving and resuming capabilities

### 📊 Comprehensive Evaluation
- **TuSimple**: Point-based accuracy (official metric)
- **CULane**: Instance-level F1-Score with 50% IoU threshold
- False Positive/False Negative rate analysis
- Qualitative assessment of integrated safety features

## Project Structure

```
.
├── preprocessing_final.ipynb                    # TuSimple dataset preprocessing
├── culane_pre_final.ipynb                      # CULane dataset preprocessing
├── training_final_main.ipynb                   # Combined dataset training
├── culane-training.ipynb                       # CULane baseline U-Net training
├── tusimple-training.ipynb                     # TuSimple baseline U-Net training
├── culane-training-attention-module.ipynb      # CULane Attention U-Net training
├── tusimple-training-attention-module.ipynb    # TuSimple Attention U-Net training
├── culane-metrics-baselineUNET.ipynb           # CULane baseline evaluation
├── tusimple-metrics-baselineUNET.ipynb         # TuSimple baseline evaluation
├── culane-metrics-attentionUNET.ipynb          # CULane attention model evaluation
├── tusimple-metrics-attentionUNET.ipynb        # TuSimple attention model evaluation
├── verify_data.ipynb                           # Data verification utilities
├── culane_verify_final.ipynb                   # CULane data verification
├── load_model.ipynb                            # Model loading and inference
├── object2_main.ipynb                          # Object detection integration
├── test4_main.ipynb                            # Testing utilities
├── annotation.ipynb                            # Annotation processing
├── training.ipynb                              # General training utilities
├── Research_Paper V2.docx                      # Full research paper
└── README.md                                   # This file
```

## Datasets

### TuSimple Dataset
- **Size**: 3,626 images (720×1280 original resolution)
- **Characteristics**: Highway driving scenes with clear weather and clean footage
- **Annotations**: Sparse (x, y) points along each lane
- **Preprocessing**: 
  - Converted point annotations to dense binary masks (5-pixel thickness)
  - Resized to 512×288 for training
  - Split: 80% train (2,900 images), 20% validation (726 images)
- **Label Files**: `label_data_0313.json`, `label_data_0531.json`, `label_data_0601.json`

### CULane Dataset
- **Size**: 88,880 training images, 9,675 validation images (590×1640 original resolution)
- **Characteristics**: Challenging scenarios including:
  - Night scenes
  - Heavy shadows
  - Busy city traffic
  - Worn-out lane markings
  - Various weather conditions
- **Annotations**: Full segmentation masks (pre-provided)
- **Preprocessing**: 
  - Resized to 320×160 for efficient training
  - Cleaned file lists into single JSON manifest

## Model Architectures

### Baseline U-Net
Classic encoder-decoder architecture serving as our baseline:
- **Encoder**: 4 downsampling blocks (64, 128, 256, 512 features)
- **Bottleneck**: 1024 features
- **Decoder**: 4 upsampling blocks with skip connections
- **Skip Connections**: Preserve spatial details while maintaining semantic context
- **Output**: Single channel binary segmentation mask

### Attention U-Net (Main Model)
Enhanced architecture with attention mechanisms:
- **Attention Gates**: Added to skip connections to filter irrelevant regions
- **Focus**: Highlights lane markings while suppressing distractions (trees, shadows, buildings)
- **Advantage**: More reliable in visually cluttered scenes
- **Performance**: 6.3% accuracy improvement on TuSimple dataset

## Integrated Safety System

### Bird's-Eye View (BEV) Transform
Both safety features rely on transforming the 2D lane prediction into a top-down view for easier geometric calculations.

### Lane Departure Warning (LDW)
- Warps predicted mask into BEV
- Fits 2nd-degree polynomial curves to left/right lane boundaries
- Averages fits over last 7 frames to reduce jitter
- Calculates vehicle position relative to lane center
- **Alert Trigger**: When drift exceeds 35% of lane width

### Adjacent Lane Obstacle Detection
- Uses pre-trained YOLOv8s model for vehicle detection
- Projects bottom-center of each detected vehicle bounding box into BEV
- Checks if projected point lies inside or outside lane boundaries
- **Obstacle Flag**: Green box for vehicles in adjacent lanes

## Training Configuration

### Hyperparameters

**TuSimple Model:**
```python
IMAGE_HEIGHT = 288
IMAGE_WIDTH = 512
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50 (Baseline: 25, Attention: 50)
OPTIMIZER = Adam
LOSS = BCEWithLogitsLoss
```

**CULane Model:**
```python
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 320
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50 (Attention: stopped at 39 due to plateau)
OPTIMIZER = Adam
LOSS = BCEWithLogitsLoss + DiceLoss
```

### Learning Rate Scheduling
- **Scheduler**: ReduceLROnPlateau
- **Mode**: Monitor validation performance
- **Factor**: 0.5 (halves learning rate)
- **Patience**: 3 epochs without improvement

### Data Augmentation (CULane)
Using Albumentations library for aggressive augmentation:
- Horizontal flip (p=0.5)
- Color jitter (brightness, contrast, saturation ±0.2)
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Loss Functions
**TuSimple**: `BCEWithLogitsLoss` (sufficient for clean highway scenes)

**CULane**: `BCEWithLogitsLoss + DiceLoss` (handles severe class imbalance)
- **BCE Loss**: Pixel-wise binary cross-entropy
- **Dice Loss**: Overlap-based metric for better segmentation of thin lane lines

## Research Paper

📄 **Full Paper**: See `Research_Paper V2.docx` for complete methodology, experiments, and analysis.

**Abstract**: This work addresses lane detection challenges in autonomous vehicles by implementing U-Net and Attention U-Net architectures on TuSimple and CULane datasets. Beyond detection, we integrate lane departure warning and adjacent lane obstacle detection using YOLOv8s. Our Attention U-Net achieved 76% accuracy on TuSimple but 36% F1-score on CULane, highlighting the need for specialized architectures and pre-trained backbones for state-of-the-art performance.

## Installation

```bash
# Clone the repository
git clone https://github.com/yavishsahrawat40/Smart-Real-Time-Lane-Detection-Departure-Warning-System-using-Deep-Learning.git
cd lane-detection-system

# Install dependencies
pip install torch torchvision
pip install albumentations opencv-python
pip install pillow numpy tqdm ultralytics
pip install jupyter notebook
```

## Usage

### 1. Data Preprocessing

**For TuSimple:**
```bash
jupyter notebook preprocessing_final.ipynb
```
- Loads raw TuSimple annotations
- Resizes images to 288x512
- Generates binary lane masks
- Saves processed data to `processed/` directory

**For CULane:**
```bash
jupyter notebook culane_pre_final.ipynb
```
- Processes CULane dataset
- Creates train/val annotations
- Generates masks from lane annotations

### 2. Training

**Single Dataset Training:**
```bash
# Train on CULane
jupyter notebook culane-training.ipynb

# Train on TuSimple
jupyter notebook tusimple-training.ipynb
```

**Combined Dataset Training:**
```bash
jupyter notebook training_final_main.ipynb
```

**Attention U-Net Training:**
```bash
# CULane with attention
jupyter notebook culane-training-attention-module.ipynb

# TuSimple with attention
jupyter notebook tusimple-training-attention-module.ipynb
```

### 3. Evaluation

```bash
# Evaluate baseline U-Net
jupyter notebook culane-metrics-baselineUNET.ipynb
jupyter notebook tusimple-metrics-baselineUNET.ipynb

# Evaluate attention U-Net
jupyter notebook culane-metrics-attentionUNET.ipynb
jupyter notebook tusimple-metrics-attentionUNET.ipynb
```

### 4. Inference

```bash
jupyter notebook load_model.ipynb
```

## Experimental Results

### TuSimple Test Set Performance (Point-Based Accuracy)

| Model | Accuracy | FP Rate | FN Rate |
|-------|----------|---------|---------|
| Baseline U-Net | 69.3% | 0.0631 | 0.0412 |
| **Attention U-Net** | **75.6%** | **0.0712** | **0.0321** |

**Key Findings:**
- Attention U-Net achieved **6.3% improvement** in accuracy
- Lower FN rate (fewer missed lanes) but slightly higher FP rate
- Attention gates effectively enhance feature selectivity on clean highway scenes

### CULane Validation Set Performance (Instance-Level F1-Score)

| Model | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| **Baseline U-Net** | **39.3%** | **36.8%** | **42.1%** |
| Attention U-Net | 36.0% | 33.1% | 39.4% |

**Key Findings:**
- Baseline U-Net performed better on challenging CULane dataset
- Attention mechanisms struggled with complex visual environments
- High false positive rate due to shadows and road artifacts being misclassified as lanes

### Comparison with State-of-the-Art

| Dataset | Our Baseline U-Net | Our Attention U-Net | SOTA (Reference Paper) |
|---------|-------------------|---------------------|------------------------|
| **TuSimple** | 69.3% | 75.6% | 96.92% |
| **CULane** | 39.3% F1 | 36.0% F1 | 68.4% F1 |

**Performance Gap Analysis:**
- 20-30+ point gap compared to specialized architectures
- SOTA methods use pre-trained backbones (e.g., ResNet-34 on ImageNet)
- Our models trained from scratch as general-purpose segmentation tools
- Specialized architectures (row-wise classification) outperform pixel-wise segmentation

### Integrated System Performance

**Lane Departure Warning:**
- ✅ Reliable performance with 7-frame averaging for stability
- ✅ Correctly triggers "Lane Departure!" alert at 35% threshold
- ⚠️ Accuracy depends on lane detection quality

**Adjacent Lane Obstacle Detection:**
- ✅ YOLOv8s accurately detects nearby vehicles
- ✅ BEV correctly flags cars in adjacent lanes as "OBSTACLE"
- ⚠️ False alerts when lane detector misclassifies shadows as lanes

### Key Observations
- **TuSimple**: Better performance due to structured highway scenarios
- **CULane**: Challenging conditions (night, shadows, worn markings) cause higher error rates
- **Attention Mechanism**: Helps on clean data but can misfocus on cluttered scenes
- **Safety Features**: Work as designed but reliability tied to lane detection accuracy

## Technical Details

### Experimental Setup
- **Platform**: Kaggle Notebook
- **GPU**: NVIDIA P100
- **Framework**: PyTorch
- **Libraries**: 
  - OpenCV for image processing
  - Ultralytics for YOLOv8s
  - Albumentations for data augmentation

### Training Time
- **CULane**: ~30 minutes per epoch (P100 GPU)
- **TuSimple**: ~3 minutes per epoch (P100 GPU)
- **Total Training**: 
  - TuSimple: 25-50 epochs
  - CULane: 39-50 epochs (early stopping applied)

### GPU Requirements
- CUDA-enabled GPU recommended (tested on NVIDIA P100)
- Mixed precision training (AMP) for memory efficiency
- Batch size adjustable based on GPU memory:
  - TuSimple: 8 (higher resolution 512×288)
  - CULane: 32 (lower resolution 320×160)

### Checkpointing
Models are saved as `.pth.tar` files containing:
- Model state dict
- Optimizer state dict
- Best validation score (Dice/Accuracy)
- Current epoch number
- Resumable training support

## Future Work

Based on our findings and research, we propose the following improvements:

### 🎯 Immediate Improvements
- **Pre-trained Backbones**: Replace randomly initialized encoder with ResNet-34 pre-trained on ImageNet to leverage learned visual features
- **Post-Processing Filters**: Add geometric and confidence thresholds to reject invalid lane predictions and reduce false positives from shadows
- **Specialized Architectures**: Implement row-wise classification approach for direct comparison with segmentation-based methods

### 🚀 Advanced Features
- **Multi-Task Learning (MTL)**: Unified model with shared backbone for:
  - Lane segmentation
  - Object detection
  - Drivable area segmentation
- **Temporal Consistency**: Video-based lane tracking for smoother predictions
- **Real-time Optimization**: Model quantization and pruning for edge deployment
- **Extended Dataset Support**: LLAMAS, BDD100K, Argoverse

### 🔬 Research Directions
- Investigate why attention mechanisms underperform on cluttered CULane scenes
- Explore hybrid approaches combining segmentation and end-to-end detection
- Study domain adaptation techniques for cross-dataset generalization

## Dependencies

### Core Requirements
```bash
Python 3.8+
PyTorch 1.9+
torchvision
CUDA Toolkit (for GPU training)
```

### Libraries
```bash
pip install torch torchvision
pip install albumentations
pip install opencv-python
pip install pillow numpy tqdm
pip install ultralytics  # For YOLOv8s
pip install jupyter notebook
```

### Full Environment
```bash
# Create conda environment
conda create -n lane_detection python=3.10
conda activate lane_detection

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install albumentations opencv-python pillow numpy tqdm ultralytics jupyter
```

## Evaluation Metrics

### TuSimple Benchmark
**Point-Based Accuracy** (Official Metric):
- A predicted point is correct if within 20 pixels horizontally of ground truth
- Formula: `Accuracy = Correct Points / Total Points`
- Also reports lane-based False Positive (FP) and False Negative (FN) rates

### CULane Benchmark
**Instance-Level F1-Score** (Official Metric):
- A predicted lane is True Positive only if IoU with ground truth > 50%
- Prevents partial detections from being counted as successful
- Reports: F1-Score, Precision, Recall based on instance-level counts

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{sahrawat2025lane,
  title={An Efficient Deep Learning Framework for Real-Time Lane Segmentation and Departure Warning},
  author={Sahrawat, Yavish and Singh, Janhvi and Gangwar, Shubham and Abhiram, G N S S},
  year={2025},
  note={Guide: Dr. Tauseef Khan}
}
```

## References

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
2. **Attention U-Net**: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", arXiv:1804.03999, 2018
3. **TuSimple Dataset**: TuSimple Lane Detection Challenge, 2017
4. **CULane Dataset**: Pan et al., "Spatial As Deep: Spatial CNN for Traffic Scene Understanding", AAAI 2018
5. **YOLOv8**: Jocher et al., "Ultralytics YOLOv8", 2023
6. **Row-wise Classification**: Yoo et al., "End-to-End Lane Marker Detection via Row-wise Classification", CVPRW 2020

For complete references, see `Research_Paper V2.docx`

## Acknowledgments

We thank:
- **Dr. Tauseef Khan** for guidance and supervision
- **TuSimple** and **CULane** teams for providing benchmark datasets
- **Kaggle** for providing free GPU resources (NVIDIA P100)
- Open-source community for PyTorch, OpenCV, and Albumentations

## License

This project is for academic and research purposes. Please cite our work if you use this code.

## Contact

**Authors:**
- Yavish Sahrawat
- Janhvi Singh
- Shubham Gangwar
- G N S S Abhiram

For questions or collaboration, please open an issue on GitHub.
