# Real-Time Panorama Stitching
C++ implementation of panorama stitching using OpenCV with performance optimizations and experimental analysis features.

## Overview

This project implements a robust panorama stitching pipeline that processes multiple images to create seamless panoramic views. The system includes comprehensive experimental analysis tools for evaluating different feature detection and matching algorithms.

### Key Capabilities
- Multi-image panorama stitching with sequential processing
- Performance-optimized with memory management and size limits
- Dual feature detector support (SIFT and ORB)
- Multiple blending methods (Linear Feathering and Simple Overlay)
- Comprehensive experimental analysis with quantitative metrics
- Batch processing of multiple datasets
- Automatic data export (CSV/JSON formats)
- Visualization tools for keypoints, matches, and performance metrics

## Features

### Core Functionality
- Feature Detection & Description: SIFT and ORB algorithms
- Feature Matching: Brute-force matcher with Lowe's ratio test  
- Homography Estimation: RANSAC-based robust estimation
- Image Warping: Perspective transformation and blending
- Multiple Blending Methods: Linear feathering and simple overlay
- Sequential Multi-Image Stitching: Processes multiple images in sequence
- Quality Control: Automatic rejection of poor-quality matches

### Performance Optimizations
- Automatic Image Resizing: Scales down large images to prevent memory overflow
- Feature Limits: Configurable maximum features for SIFT (800) and ORB (500)
- Panorama Size Limits: Maximum panorama dimensions to prevent crashes  
- Progressive Compression: Dynamic compression during multi-image stitching
- Memory Management: Automatic cleanup and early termination
- Match Quality Filtering: Rejects homographies with poor inlier ratios

## How It Works - Technical Deep Dive

### Algorithm Overview

The panorama stitching process follows a classical computer vision pipeline with modern optimizations:

```
Input Images → Feature Detection → Feature Matching → Homography Estimation → 
Image Warping → Blending → Quality Control → Progressive Compression → Output
```

### Sequential Stitching Process

For multiple images (N > 2), the algorithm works sequentially:

1. Initialize: Start with first two images
2. Stitch: Create initial panorama from images 1 + 2
3. Iterate: For each remaining image (3, 4, 5...):
   - Detect features in current panorama and new image
   - Match features between panorama and new image
   - Estimate homography transformation
   - Quality check: Skip image if transformation quality is poor
   - Warp new image to align with panorama
   - Blend images together
   - Compress if needed to manage memory
4. Final: Export final panorama and analysis data

### Experimental Analysis Tools
- [x] Data Export: CSV and JSON export of experimental results
- [x] Keypoint & Match Visualization: Visual overlay of detected features and matches
- [x] Histogram Plotting: Distribution analysis of match distances
- [x] Quantitative Metrics Tables: Comprehensive performance statistics

## Installation

### Prerequisites
1. Install vcpkg and integrate with your build system
2. Install OpenCV via vcpkg:
   ```bash
   vcpkg install opencv[contrib]:x64-windows
   ```

### Build Process
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Debug
```

## Usage

The program supports three different execution modes:

### 1. Batch Process All Datasets
```bash
./PanoramaStitching.exe
```

### 2. Process Single Dataset
```bash
./PanoramaStitching.exe data/indoor
./PanoramaStitching.exe data/outdoor
./PanoramaStitching.exe data/wall
```

### 3. Process Two Images
```bash
./PanoramaStitching.exe image1.jpg image2.jpg
./PanoramaStitching.exe data/indoor/indoor1-min.jpg data/indoor/indoor2-min.jpg
```

### Input Image Requirements
- Format: JPEG, PNG, or other OpenCV-supported formats
- Size: Images are automatically resized if larger than 1500px in any dimension to save memory
- Overlap: Images should have sufficient overlap (recommended: 20-40%)

## Configuration

Key parameters can be modified in main.cpp:

```cpp
// Feature Detection Configuration  
const bool USE_SIFT = true;              // true for SIFT, false for ORB
const bool USE_FEATHERING = true;        // true for linear feathering, false for overlay
const double RATIO_THRESHOLD = 0.8;      // Lowe's ratio test threshold
const double RANSAC_THRESHOLD = 5.0;     // RANSAC reprojection threshold

// Quality Control Thresholds
const int MIN_GOOD_MATCHES = 20;         // Minimum good matches required
const double MIN_INLIER_RATIO = 0.25;    // Minimum inlier ratio (25%)
const int MIN_ABSOLUTE_INLIERS = 8;      // Minimum absolute number of inliers

// Performance Limits
const int MAX_SIFT_FEATURES = 800;       // Maximum SIFT features
const int MAX_ORB_FEATURES = 500;        // Maximum ORB features
const double COMPRESSION_THRESHOLD = 0.6; // Compress at 60% of max size (dynamic)
```

## Experimental Analysis

When RUN_EXPERIMENTS = true, the system performs comprehensive analysis with:

### Analysis Parameters
- Feature Detectors: SIFT and ORB
- Blending Methods: Linear Feathering and Simple Overlay  
- RANSAC Thresholds: 1.0, 3.0, 5.0, 7.0, 10.0 pixels
- Metrics Collected: Timing, keypoints, matches, inliers, ratios

### Generated Outputs
1. CSV Data: [dataset]_pair_1_2_experimental_data.csv
2. JSON Data: [dataset]_pair_1_2_experimental_data.json  
3. Match Visualizations: [dataset]_pair_1_2_[detector]_[blend]_matches.jpg
4. Distance Histograms: [dataset]_pair_1_2_[detector]_[blend]_distances.jpg
5. Console Tables: Real-time quantitative metrics display

### Metrics Included
- Keypoints detected (both images)
- Good matches after Lowe's ratio test
- RANSAC inliers count and ratio
- Timing breakdown (detection, matching, RANSAC, total)
- Average match distance
- Panorama dimensions

## Output Structure

```
output/
├── indoor/
│   ├── indoor_panorama.jpg
│   └── experimental_analysis/
│       ├── indoor_pair_1_2_experimental_data.csv
│       ├── indoor_pair_1_2_experimental_data.json
│       ├── indoor_pair_1_2_SIFT_Feathering_matches.jpg
│       ├── indoor_pair_1_2_SIFT_Feathering_distances.jpg
│       ├── indoor_pair_1_2_ORB_Feathering_matches.jpg
│       ├── indoor_pair_1_2_ORB_Feathering_distances.jpg
│       └── [additional visualization files...]
├── outdoor/
│   ├── outdoor_panorama.jpg
│   └── experimental_analysis/
│       └── [similar experimental files...]
└── wall/
    ├── wall_panorama.jpg
    └── experimental_analysis/
        └── [similar experimental files...]
```

## Performance Characteristics

### Processing Capabilities
- Memory-optimized: Automatic image resizing prevents crashes
- Speed-optimized: Feature limits balance quality vs processing time
- Scalable: Handles complete datasets with dynamic compression
- Quality-controlled: Rejects poor matches to prevent distorted results
- Robust: Continues processing even when individual images fail

### Performance Characteristics
- Feature Detection: SIFT slower but higher quality than ORB
- Image Size: Processing time scales with input resolution
- Match Quality: Repetitive textures cause matching failures  
- Memory Usage: Dynamic compression prevents memory overflow
- Quality Control: Prevents distorted output by rejecting poor matches

### Match Quality Analysis

Indoor Dataset (Good Quality with One Failure)
- Good matches: 58-244 per image pair
- Inlier ratios: 12.1% - 48.4% (mixed quality)
- Issue: Indoor5 rejected with 12.1% inlier ratio (below 25% threshold)
- Result: High-quality panorama from 4 images, 5th image rightfully rejected

Outdoor Dataset (Excellent Quality)  
- Good matches: 50-334 per image pair
- Inlier ratios: 48.0% - 80.8% (good to excellent)
- Issue: Outdoor7 failed due to size limits (7877x10005 pixels), not quality
- Result: Excellent panorama from 6 images with superior alignment

Wall Dataset (Poor Quality - Case Study)
- Good matches: 20-147 per image pair
- Inlier ratios: 7.5% - 31.3% (mostly poor, few acceptable)
- Problem: Repetitive texture patterns cause false matches
- Solution: Quality control system rejects poor matches (< 25% inlier ratio)
- Result: Only 2 images processed to maintain quality

## Technical Implementation

### Architecture Overview
```
Input Images → Feature Detection → Feature Matching → Homography Estimation → Image Warping → Blending → Output Panorama
              ↓                    ↓                   ↓                        ↓             ↓
         (SIFT/ORB)           (Lowe's Ratio)      (RANSAC)            (Perspective Transform) (Feathering/Overlay)
```

### Key Algorithms
1. SIFT Features: Scale-invariant detection with 128-dimensional descriptors
2. ORB Features: Fast binary descriptors with rotation invariance
3. Brute-Force Matching: Exhaustive search with Lowe's ratio test (threshold: 0.8)
4. RANSAC Homography: Robust estimation with configurable pixel thresholds
5. Linear Feathering: Smooth blending based on distance from image boundaries

### Data Structures
- FeatureMatchingResults: Stores keypoints, matches, and timing data
- HomographyResults: Contains homography matrix and inlier information  
- ExperimentalResults: Comprehensive metrics for analysis export

### Performance Functions
- resizeImageIfNeeded(): Scales large input images to manageable sizes
- compressPanoramaIfNeeded(): Progressive compression during multi-image stitching
- isPanoramaTooLarge(): Safety checks to prevent memory overflow

## Known Limitations

### Current Constraints
1. Sequential Stitching: Only supports linear image sequences (not full 360° panoramas)
2. Memory Limits: Large images require automatic downscaling  
3. Panorama Size: Hard limits prevent very large panorama generation
4. Feature Matching: Relies on sufficient texture and overlap between images
5. Planar Assumption: Works best with roughly planar scenes
6. Repetitive Textures: Quality control rejects scenes with ambiguous features

### Dataset-Specific Results
- Indoor: Successfully processes 5/5 images (100%) with good quality
- Outdoor: Processes 6/7 images (86%) with excellent alignment
- Wall: Processes 2/6 images (33%) - quality control prevents distorted output


## Project Structure
```
rt_panorama_stitching/
├── CMakeLists.txt           # Build configuration
├── main.cpp                 # Main application entry point (827 lines)
├── include/                 # Header files
├── src/                     # Implementation files
├── data/                    # Test image datasets
│   ├── indoor/             # Indoor scene images (5 images)
│   ├── outdoor/            # Outdoor scene images (7 images)
│   └── wall/               # Wall/texture images (6 images)
├── output/                 # Results and generated files
│   ├── indoor/             # Indoor dataset results
│   ├── outdoor/            # Outdoor dataset results
│   └── wall/               # Wall dataset results
└── build/                  # CMake build directory
```