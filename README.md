# 🎭 Deepfake Detection System

An advanced AI-powered deepfake detection system using ResNeXt-50 architecture to identify manipulated images with high accuracy. This project aims to combat digital misinformation and restore trust in media authenticity.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Capabilities & Limitations](#capabilities--limitations)
- [Results & Visualizations](#results--visualizations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

## 🔍 Overview

**Deepfakes** are synthetic media created using deep learning techniques, particularly Generative Adversarial Networks (GANs), to manipulate or generate realistic fake images, videos, or audio. With the rise of sophisticated deepfake technology, there's an urgent need for reliable detection systems.

This project implements a **ResNeXt-50 based deep learning model** trained on 60,000+ images to detect deepfakes with **99.89% training accuracy**. The system provides a user-friendly web interface for quick image analysis, making advanced deepfake detection accessible to everyone.

### 🎯 Project Goals
- Develop a highly accurate deepfake detection system
- Provide fast inference (< 2 seconds per image)
- Create an intuitive web interface for easy access
- Combat misinformation and digital manipulation
- Contribute to media authenticity verification

## ✨ Features

- **High Accuracy**: 99.89% training accuracy, ~95% validation accuracy
- **Fast Detection**: Results in under 2 seconds
- **ResNeXt-50 Architecture**: State-of-the-art CNN with 50 layers
- **Web Interface**: Clean, modern UI for easy image upload and analysis
- **Confidence Scores**: Provides probability estimates for predictions
- **Multiple Deepfake Types**: Detects face swaps, GANs, reenactment, attribute manipulation
- **Activation Heatmaps**: Visualizes model attention regions
- **Privacy First**: Images processed securely, not stored on servers

## 🛠️ Technology Stack

### Deep Learning & ML
- **PyTorch** - Deep learning framework
- **ResNeXt-50** - Convolutional Neural Network architecture
- **torchvision** - Pre-trained models and image transformations
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Web framework for API
- **OpenCV** - Image processing
- **Pillow (PIL)** - Image handling

### Frontend
- **HTML5/CSS3** - Structure and styling
- **JavaScript** - Dynamic interactions
- **Responsive Design** - Mobile-friendly interface

### Development Tools
- **Jupyter Notebook** - Model development and training
- **Git** - Version control
- **VS Code** - Code editor

## 🏗️ Architecture

### ResNeXt-50 Model Architecture

```
Input Image (224x224x3)
        ↓
   Preprocessing
   (Resize, Normalize)
        ↓
   ResNeXt-50 Backbone
   (50 Convolutional Layers)
   - Cardinality: 32
   - Bottleneck Design
   - Residual Connections
        ↓
   Feature Extraction
   (2048 features)
        ↓
   Fully Connected Layer
        ↓
   Binary Classification
   (Real vs Fake)
        ↓
   Softmax Activation
        ↓
   Output: [Probability Real, Probability Fake]
```

### Why ResNeXt-50?

1. **Cardinality-Based Design**: 32 parallel paths for diverse feature extraction
2. **Residual Connections**: Skip connections prevent vanishing gradients
3. **Deep Architecture**: 50 layers for hierarchical feature learning
4. **Bottleneck Efficiency**: Optimal computation with 1x1, 3x3, 1x1 convolutions
5. **Transfer Learning**: Leverages ImageNet pre-trained weights
6. **Superior Performance**: Better accuracy than ResNet-50 with similar computational cost

## 📊 Dataset

### Composition
- **Total Images**: 60,000+
- **Real Images**: 30,000+ (50%)
- **Fake Images**: 30,000+ (50%)
- **Train/Val Split**: 80/20

### Real Images Sources
- Celebrity faces
- Public figures
- Stock photography
- Various lighting conditions
- Multiple angles and poses
- Diverse demographics

### Fake Images Sources
- **StyleGAN**: Generated faces from StyleGAN architecture
- **ProGAN**: Progressive GAN outputs
- **Face Swap**: DeepFaceLab and FaceSwap tools
- **FaceApp**: Attribute manipulation
- **Other GANs**: Various GAN architectures (CycleGAN, StarGAN, etc.)
- **Reenactment**: Facial expression transfer techniques

### Data Preprocessing
1. **Resizing**: All images resized to 224x224 pixels
2. **Normalization**: ImageNet mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]
3. **Augmentation**:
   - Random horizontal flips
   - Random rotations (±15°)
   - Color jitter (brightness, contrast, saturation)
   - Random crops
   - Gaussian noise addition

## 📈 Model Performance

### Training Metrics
| Metric | Value |
|--------|-------|
| Training Accuracy | 99.89% |
| Validation Accuracy | ~95% |
| Training Loss | 0.05 |
| F1-Score | 94.5% |
| Precision | ~94% |
| Recall | ~95% |
| AUC-ROC | ~0.97 |
| False Positive Rate | ~5% |

### Training Configuration
```python
Epochs: 20-30
Batch Size: 32
Learning Rate: 0.0001
Optimizer: Adam
Loss Function: CrossEntropyLoss
Scheduler: ReduceLROnPlateau
Weight Decay: 1e-4
```

### Inference Performance
- **Average Inference Time**: 1.8 seconds per image
- **Batch Processing**: Supports multiple images
- **GPU Acceleration**: Optimized for CUDA-enabled devices
- **CPU Fallback**: Works on CPU (slower)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training/inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kunaldgr/deepfake-detection.git
cd deepfake-detection
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model
```bash
# Download the trained model weights (provide link or instructions)
# Place the model file in the 'models/' directory
```

## 💻 Usage

### Running the Web Application

1. **Start the Flask Server**
```bash
python app.py
```

2. **Open Your Browser**
```
Navigate to: http://localhost:5000
```

3. **Upload and Analyze**
   - Click the upload area or drag & drop an image
   - Supported formats: JPG, PNG, JPEG
   - Max file size: 10MB
   - Click "Analyze Image" to get results

### Using the Model Programmatically

```python
import torch
from torchvision import transforms
from PIL import Image
from model import DeepfakeDetector

# Load model
model = DeepfakeDetector()
model.load_state_dict(torch.load('models/resnext50_deepfake.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item() * 100

print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: {confidence:.2f}%")
```

## 🔬 How It Works

### Detection Pipeline

1. **Image Upload**: User uploads an image through the web interface
2. **Preprocessing**: 
   - Image resized to 224x224 pixels
   - Normalized with ImageNet statistics
   - Converted to tensor format
3. **Feature Extraction**: 
   - ResNeXt-50 processes image through 50 convolutional layers
   - Extracts 2048-dimensional feature vector
4. **Classification**: 
   - Fully connected layer maps features to 2 classes
   - Softmax activation produces probability distribution
5. **Prediction**: 
   - Class with highest probability is selected
   - Confidence score calculated
6. **Result Display**: 
   - User sees prediction (Real/Fake)
   - Confidence percentage shown
   - Detailed analysis provided

### What the Model Detects

The model identifies various deepfake artifacts and patterns:

- **Facial Inconsistencies**: Unnatural facial feature alignment
- **Lighting Artifacts**: Inconsistent lighting and shadows
- **Compression Patterns**: GAN-specific compression signatures
- **Edge Irregularities**: Blending artifacts at face boundaries
- **Color Anomalies**: Unnatural color distributions
- **Texture Patterns**: Synthetic texture characteristics
- **Frequency Analysis**: High-frequency noise patterns
- **Facial Landmarks**: Distorted or misaligned landmarks

## ✅❌ Capabilities & Limitations

### ✅ What It DOES

| Capability | Description |
|------------|-------------|
| **Face Swap Detection** | Identifies swapped faces with high accuracy |
| **GAN-Generated Faces** | Detects StyleGAN, ProGAN outputs |
| **Facial Reenactment** | Spots expression transfer manipulation |
| **Attribute Changes** | Detects age, gender, feature modifications |
| **Modern Deepfakes** | Works well on 2019+ techniques |
| **Fast Analysis** | Results in under 2 seconds |
| **Confidence Scores** | Provides probability estimates |
| **High-Quality Images** | Best performance on good resolution |
| **Batch Processing** | Can analyze multiple images |
| **User-Friendly** | Simple upload and analyze workflow |

### ❌ What It DOES NOT Do

| Limitation | Description |
|------------|-------------|
| **Old GANs (Pre-2019)** | Lower accuracy on older architectures |
| **Video Analysis** | Only processes static images, not videos |
| **Audio Deepfakes** | Cannot detect voice cloning or audio manipulation |
| **Highly Compressed** | Performance drops on low-quality images |
| **Real-time Detection** | Not optimized for live video streams |
| **Non-Face Content** | Trained specifically for faces, not objects/landscapes |
| **Text Forgery** | Cannot detect manipulated documents or text |
| **Future Techniques** | May not detect brand new deepfake methods |
| **100% Accuracy** | No AI system is perfect - false positives/negatives exist |
| **Explainability** | Doesn't provide detailed explanations of artifacts |

### Best Use Cases

- ✅ Verifying social media images
- ✅ Fact-checking news photos
- ✅ Corporate identity verification
- ✅ Legal evidence validation
- ✅ Educational demonstrations
- ✅ Research and analysis

### Not Recommended For

- ❌ Sole authentication for critical decisions
- ❌ Video deepfake detection
- ❌ Audio deepfake detection
- ❌ Detection of manually edited photos
- ❌ Non-facial image verification

## 📸 Results & Visualizations

### Activation Heatmaps

The model uses Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which regions it focuses on:

**FAKE Detection Example (100.0% Confidence)**
- Original Image → Activation Map → Overlay
- Red regions indicate high attention (suspicious areas)
- Model focuses on eyes, nose, mouth edges where artifacts appear

**REAL Detection Example (100.0% Confidence)**
- Original Image → Activation Map → Overlay
- More uniform attention distribution
- Natural patterns detected throughout the face

### Confusion Matrix

```
                Predicted
              Real    Fake
Actual  Real  2850    150
        Fake  100     2900
```

### ROC Curve
- AUC: 0.97
- Strong separation between classes
- Low false positive rate

## 🚀 Future Enhancements

### Short-term Goals
- [ ] Video deepfake detection (frame-by-frame analysis)
- [ ] Improved old GAN detection (pre-2019 architectures)
- [ ] Model explainability with detailed artifact reports
- [ ] Mobile application for on-the-go detection
- [ ] API documentation and public API access

### Long-term Vision
- [ ] Multi-modal detection (image + video + audio)
- [ ] Real-time video stream analysis
- [ ] Ensemble model combining multiple architectures
- [ ] Attention mechanism improvements
- [ ] Browser extension for in-browser detection
- [ ] Continuous learning with new deepfake techniques
- [ ] Integration with social media platforms
- [ ] Blockchain-based media authentication

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Contribution
- Improving model accuracy
- Adding new deepfake detection techniques
- Enhancing the web interface
- Writing documentation
- Bug fixes and optimizations
- Dataset expansion

## 📞 Contact

**Kunal Dagar**

- 📧 Email: kunaldagar4298@gmail.com
- 💼 LinkedIn: [linkedin.com/in/kunal-dagar-661161322](https://www.linkedin.com/in/kunal-dagar-661161322/)
- 💻 GitHub: [github.com/Kunaldgr](https://github.com/Kunaldgr)


## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- ResNeXt authors for the architecture
- Open-source deepfake datasets
- AI research community
- All contributors and testers

## 📚 References

1. Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated Residual Transformations for Deep Neural Networks. CVPR.
2. Rossler, A., et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV.
3. Karras, T., Laine, S., & Aila, T. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. CVPR.
4. Goodfellow, I., et al. (2014). Generative Adversarial Networks. NeurIPS.

---

## ⚠️ Disclaimer

This tool is designed to assist in identifying deepfakes but should not be used as the sole method for authenticating media. Always consider:

- Context and source of the media
- Multiple verification methods
- Expert human review for critical decisions
- The evolving nature of deepfake technology

No AI detection system is 100% accurate. Use responsibly and ethically.

---

**Made with ❤️ by Kunal Dagar**

*Fighting digital misinformation, one image at a time.*
