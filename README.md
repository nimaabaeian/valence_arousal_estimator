# Valence/Arousal Estimator
## Real-Time Emotion Estimation using EmoNet

A YARP RFModule for real-time continuous valence and arousal estimation from detected faces using the EmoNet pretrained model.

---

## 📖 Overview

This module provides a YARP interface to the **EmoNet** model, enabling real-time estimation of emotional valence (pleasantness) and arousal (intensity) from face images. It processes face detections from a face recognition module and outputs continuous emotion values.

### Reference Paper
> **"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"**  
> Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, Maja Pantic  
> _Nature Machine Intelligence_, January 2021  
> https://www.nature.com/articles/s42256-020-00280-0

**Original Repository:** https://github.com/face-analysis/emonet

---

## ✨ Features

- ⚡ Real-time valence and arousal estimation via YARP
- 🔄 Batch processing of multiple faces per frame
- 🚀 GPU acceleration support (CUDA)
- 🎯 Configurable face detection score threshold
- ⏱️ Timestamped output messages
- 🖼️ Efficient YARP-to-NumPy image conversion

---

## 🔌 YARP Interface

### Input Ports

#### 1. `/emonet/faceID/annotations:i`
- **Type:** `Bottle`
- **Description:** Face detections with bounding boxes and labels
- **Format:**
  ```
  ((class "PersonName") (score 0.95) (box (x1 y1 x2 y2)))
  ((class "AnotherPerson") (score 0.87) (box (x1 y1 x2 y2)))
  ```
- **Fields:**
  - `class`: Face label/name (string)
  - `score`: Detection confidence [0.0 - 1.0] (float)
  - `box`: Bounding box coordinates [x1, y1, x2, y2] (integers)

#### 2. `/emonet/webcam:i`
- **Type:** `ImageRgb`
- **Description:** RGB image stream (same frame as face detections)
- **Requirements:** Must be synchronized with face annotations

### Output Port

#### `/emonet/valence_arousal:o`
- **Type:** `Bottle` (with timestamp envelope)
- **Description:** Valence and arousal estimates for each detected face
- **Format:**
  ```
  "PersonName1" <valence1> <arousal1> "PersonName2" <valence2> <arousal2> ...
  ```
- **Structure:** Repeating triplets of:
  1. **Name** (string): Face label from input
  2. **Valence** (float64): Emotional pleasantness [-1.0, 1.0]
     - `-1.0`: Very negative (sad, angry)
     - `+1.0`: Very positive (happy, joyful)
  3. **Arousal** (float64): Emotional intensity [-1.0, 1.0]
     - `-1.0`: Very calm (relaxed, sleepy)
     - `+1.0`: Very excited (surprised, alert)

**Example Output:**
```
"Alice" 0.42 0.31 "Bob" -0.15 0.68 "Charlie" 0.78 -0.22
```
- Alice: Moderately positive and slightly aroused
- Bob: Slightly negative but quite aroused
- Charlie: Very positive but calm

**Timestamp:** Each message includes a YARP timestamp envelope for synchronization.

---

## 🛠️ Installation

### Prerequisites
1. **YARP** with Python bindings ([installation guide](https://www.yarp.it/latest/install.html))
2. **Python 3.10+**
3. **Conda** (Miniconda or Anaconda)

### Build Instructions

1. Clone and navigate to the repository:
```bash
cd /path/to/valence_arousal_estimator
mkdir build && cd build
```

2. Configure with CMake:
```bash
cmake ..
```

3. Build (creates conda environment automatically):
```bash
make
```

4. Install:
```bash
make install
```

The build process automatically creates a conda environment with all dependencies (PyTorch, CUDA toolkit, OpenCV, etc.).

---

## 🚀 Usage

### Basic Usage
```bash
valenceArousalMapper
```

### With Custom Parameters
```bash
valenceArousalMapper --model_path /path/to/emonet_8.pth --device cuda --min_score 0.7
```

### Direct Python Invocation
```bash
python modules/valence_arousal_estimator/valence_arousal_estimator.py \
    --nclasses 8 \
    --device cuda \
    --period 0.033 \
    --min_score 0.5
```

---

## ⚙️ Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | string | _(auto-detect)_ | Path to pretrained .pth model file |
| `--nclasses` | int | `8` | Number of emotion classes: `5` or `8` |
| `--device` | string | `cuda` | Computation device: `cuda` or `cpu` |
| `--input_size` | int | `256` | Face crop resize dimension (pixels) |
| `--period` | float | `0.033` | Module update period in seconds (~30 FPS) |
| `--min_score` | float | `0.5` | Minimum face detection confidence threshold [0.0-1.0] |
| `--emonet_root` | string | `""` | Path to emonet repository (if not in PYTHONPATH) |

---

## 🧠 Pretrained Models

Two models are available in the `pretrained/` directory:

### **emonet_8.pth** (Default)
- **Emotion Classes:** 8 (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
- **Performance (AffectNet test set):**
  - Valence: CCC=0.82, RMSE=0.29
  - Arousal: CCC=0.75, RMSE=0.27

### **emonet_5.pth**
- **Emotion Classes:** 5 (Neutral, Happy, Sad, Surprise, Fear)
- **Performance (AffectNet test set):**
  - Valence: CCC=0.90, RMSE=0.24
  - Arousal: CCC=0.80, RMSE=0.24

Both models output continuous valence and arousal values in addition to discrete emotion classifications.

---

## 📁 Repository Structure

```
valence_arousal_estimator/
├── CMakeLists.txt
├── README.md
├── LICENSE.txt
├── modules/
│   ├── CMakeLists.txt
│   ├── environment.yml                    # Conda environment specification
│   └── valence_arousal_estimator/
│       ├── CMakeLists.txt
│       ├── valence_arousal_estimator.py   # Main YARP RFModule
│       ├── emonet/
│       │   ├── __init__.py
│       │   └── models/
│       │       ├── __init__.py
│       │       └── emonet.py              # EmoNet model architecture
│       └── pretrained/
│           ├── emonet_5.pth               # 5-class model weights
│           └── emonet_8.pth               # 8-class model weights
└── app/
    └── valence_arousal_estimator/
        ├── conf/
        │   └── valence_arousal_estimator.ini
        └── scripts/
            └── valence_arousal_estimator.xml
```

---

## 🔗 Complete Example Setup
---

## 🐛 Troubleshooting

### Issue: "YARP network not available"
- Ensure `yarpserver` is running
- Check network configuration: `yarp detect`

### Issue: "Model file not found"
- Verify model exists in `modules/valence_arousal_estimator/pretrained/`
- Specify explicit path: `--model_path /full/path/to/emonet_8.pth`

### Issue: "CUDA out of memory"
- Switch to CPU: `--device cpu`
- Reduce batch size (fewer simultaneous faces)
- Use smaller input size: `--input_size 128`

### Issue: "No faces detected"
- Lower detection threshold: `--min_score 0.3`
- Verify face detection module is running and publishing
- Check port connections: `yarp name list`

---

## 📄 License

This code is available under a **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND)**.

See [LICENSE.txt](LICENSE.txt) for full details.

---

## 📚 Citation

If you use this code or the EmoNet model, please cite:

```bibtex
@article{toisoul2021estimation,
  author  = {Antoine Toisoul and Jean Kossaifi and Adrian Bulat and 
             Georgios Tzimiropoulos and Maja Pantic},
  title   = {Estimation of continuous valence and arousal levels from faces 
             in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  volume  = {3},
  pages   = {42--50},
  doi     = {10.1038/s42256-020-00280-0},
  url     = {https://www.nature.com/articles/s42256-020-00280-0}
}
```

---

## 👥 Credits

- **Original EmoNet Model:** [face-analysis/emonet](https://github.com/face-analysis/emonet)
- **Authors:** Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, Maja Pantic
- **Organizations:** Samsung AI Center Cambridge, Imperial College London
- **YARP Integration:** This repository

**Full Paper (View-Only):** https://rdcu.be/cdnWi

---

## 📞 Support

For issues related to:
- **YARP integration:** Open an issue in this repository
- **EmoNet model:** Refer to the [original repository](https://github.com/face-analysis/emonet)
- **YARP framework:** Visit [YARP documentation](https://www.yarp.it)o /yourApp/emotions:i
```

---

## 📊 Understanding the Output

### Valence-Arousal Space

```
        High Arousal (+1.0)
              ↑
    Angry  Excited  Joyful
       ↖     ↑     ↗
         ＼   |   ／
Negative ←---+---→ Positive
(-1.0)   ／   |   ＼    (+1.0)
       ↙     ↓     ↘
    Sad   Calm   Relaxed
              ↓
        Low Arousal (-1.0)
```

**Examples:**
- **Happy:** valence = +0.8, arousal = +0.6
- **Angry:** valence = -0.7, arousal = +0.9
- **Sad:** valence = -0.6, arousal = -0.3
- **Relaxed:** valence = +0.3, arousal = -0.8

## License

This code is available under a **Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND)**.

See [LICENSE.txt](LICENSE.txt) for full details.

## Citation

If you use this code, please cite:

```bibtex
@article{toisoul2021estimation,
  author  = {Antoine Toisoul and Jean Kossaifi and Adrian Bulat and Georgios Tzimiropoulos and Maja Pantic},
  title   = {Estimation of continuous valence and arousal levels from faces in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  url     = {https://www.nature.com/articles/s42256-020-00280-0}
}
```

## Credits

- **Original EmoNet Model:** https://github.com/face-analysis/emonet
- **Authors:** Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, Maja Pantic
- **YARP Module Adaptation:** This repository

---

**Full paper (view-only):** https://rdcu.be/cdnWi

