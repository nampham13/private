# Testing the Preprocessing Pipeline with Your Circuit Board Image

## Quick Start Guide

### Method 1: Simple Quick Test

1. **Save your circuit board image** as `circuit_board.jpg` in the `examples/` folder
2. **Run the quick test:**
   ```bash
   cd examples/
   python quick_test.py
   ```

This will automatically:
- ✅ Load your circuit board image
- ✅ Apply bilateral filtering (noise reduction + edge preservation)
- ✅ Enhance contrast using HSV histogram equalization
- ✅ Generate depth map using edge detection
- ✅ Create 4-channel RGB-D tensor
- ✅ Show comprehensive visualizations
- ✅ Save results as `.npz` file

### Method 2: Advanced Processing

For more detailed analysis, use the comprehensive processor:

```bash
cd examples/
python circuit_board_processor.py
```

This provides:
- Multiple filtering options (Gaussian, Median, Bilateral)
- Different equalization methods (YUV, HSV, LAB)
- Advanced depth analysis with 3D visualization
- Detailed component detection statistics

### Method 3: Command Line Interface

Using the CLI tool:

```bash
cd tools/
python cli.py ../examples/circuit_board.jpg output.npz --preset high_quality --visualize
```

## Expected Results for Circuit Board Images

### 🔍 What the Pipeline Does:

1. **Image Filtering**:
   - Removes sensor noise while preserving component edges
   - Bilateral filtering works best for electronics (preserves fine details)

2. **Contrast Enhancement**:
   - Makes components more distinguishable from PCB background
   - HSV equalization enhances component visibility

3. **Depth Generation**:
   - Components appear "closer" (higher depth values)
   - PCB traces have medium depth
   - Background PCB has lower depth values

4. **RGB-D Output**:
   - 4-channel tensor: Red, Green, Blue, Depth
   - Ready for deep learning models
   - Normalized values [0, 1]

### 📊 Typical Results for Electronics:

- **Components** (resistors, capacitors, ICs): High depth (0.7-1.0)
- **Copper traces**: Medium depth (0.3-0.7)  
- **PCB substrate**: Low depth (0.0-0.3)
- **Solder joints**: Variable depth based on height

### 📁 Output Files:

After processing, you'll get:
- `circuit_board_rgbd.npz` - Main RGB-D tensor file
- `depth_map.png` - Depth visualization
- Visualization plots showing all processing steps

## Troubleshooting

### If the script doesn't find your image:
1. Make sure the image is named `circuit_board.jpg` or `circuit_board.png`
2. Place it in the `examples/` directory
3. The script also auto-detects other image files

### If you get import errors:
1. Run from the correct directory: `cd examples/`
2. Install missing dependencies: `pip install opencv-python matplotlib numpy`

### For best results with circuit boards:
- Use high-resolution images (>512x512)
- Ensure good lighting without shadows
- Avoid reflections on shiny components

## Understanding Your Results

The visualizations will show:
1. **Original vs Processed**: How filtering improves the image
2. **Depth Map**: Components should appear as bright regions
3. **Channel Analysis**: Statistics for each RGB-D channel
4. **Depth Distribution**: Histogram showing depth value spread

Your circuit board's depth map should clearly distinguish:
- Electronic components (bright/high depth)
- Circuit traces (medium depth)
- PCB background (dark/low depth)

This RGB-D data can then be used for:
- 3D object detection
- Component classification
- Defect detection
- Assembly verification
- Quality control applications
