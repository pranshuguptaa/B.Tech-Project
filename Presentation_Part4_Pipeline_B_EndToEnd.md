# MULTIMODAL STRESS DETECTION SYSTEM
## Presentation Content - Part 4: Pipeline B (End-to-End Deep Learning)

---

## SLIDE 10: METHODOLOGY IV - PIPELINE B (END-TO-END)
### Part 1: Feature Representation

### YOLOV11N FOR FACIAL EMOTION DETECTION

**What is YOLOv11?**

**YOLO = You Only Look Once**
- State-of-the-art real-time object detection framework
- Single-pass architecture (processes entire image once)
- Simultaneously predicts bounding boxes AND class labels
- YOLOv11 = Latest version (2023-2024) with improved accuracy and speed

**Why YOLOv11-nano (yolo11n)?**

**Model Variants:**
- **YOLOv11n (nano):** Smallest, fastest (5.6 MB, 2.6M parameters)
- **YOLOv11s (small):** Balanced (9.4 MB)
- **YOLOv11m (medium):** Higher accuracy (20 MB)
- **YOLOv11l (large):** Best accuracy (25 MB)
- **YOLOv11x (extra-large):** Maximum accuracy (30+ MB)

**Our Choice: YOLOv11n**
✅ **Speed:** 100+ FPS on GPU, 20-30 FPS on CPU  
✅ **Size:** 5.6 MB - deployable on edge devices  
✅ **Accuracy:** Sufficient for controlled RAVDESS dataset  
✅ **Real-time Capability:** Suitable for live stress monitoring  

---

### YOLO TRAINING PROCESS

**Transfer Learning Approach:**

```
Pre-trained YOLOv11n (COCO dataset)
    ↓
Fine-tune on Custom Facial Emotion Dataset
    ↓
Detect faces + Classify emotions (8 classes)
    ↓
Map emotions to stress labels (same mapping as Pipeline A)
```

**Training Configuration:**

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Load pre-trained weights

model.train(
    data="data.yaml",           # Dataset configuration
    epochs=80,                  # Training iterations
    imgsz=640,                  # Input image size (640×640)
    batch=8,                    # Batch size
    device=0,                   # GPU 0 (or "cpu")
    amp=True                    # Automatic Mixed Precision (faster)
)
```

**Dataset Configuration (data.yaml):**
```yaml
# Custom facial emotion dataset
path: /path/to/facial_emotion_dataset
train: train/images
val: valid/images
test: test/images

# Class names (8 emotions)
names:
  0: neutral
  1: calm
  2: happy
  3: sad
  4: angry
  5: fearful
  6: disgust
  7: surprised
```

**Training Details:**
- **Epochs:** 80 (more than Pipeline A due to larger model capacity)
- **Batch Size:** 8 (limited by GPU memory)
- **Image Size:** 640×640 (YOLO standard)
- **Augmentation:** Built-in (random flip, scale, crop, color jitter)
- **Optimizer:** SGD with momentum (YOLO default)
- **Learning Rate:** Adaptive (starts at 0.01, decays)

**Training Time:**
- **GPU (NVIDIA RTX 3060):** ~2-3 hours for 80 epochs
- **CPU:** ~15-20 hours (not recommended)

**Model Output:**
- **File:** `yolo11n.pt` (fine-tuned weights)
- **Size:** 5.6 MB
- **Format:** PyTorch (.pt) model

---

### YOLO INFERENCE & PREDICTION

**Inference Pipeline:**

```
Input: Video frame (1920×1080)
    ↓
Resize to 640×640 (YOLO input size)
    ↓
YOLOv11n Forward Pass
    ↓
Output: [Bounding Box, Confidence, Class Probabilities]
    ↓
Extract emotion class (0-7)
    ↓
Map to stress label (0 or 1)
```

**Output Format:**
```python
# Example YOLO output for one frame
{
    'boxes': [[x1, y1, x2, y2]],        # Face bounding box
    'confidence': [0.95],                # Detection confidence
    'class': [4],                        # Emotion class (4 = angry)
    'class_probs': [0.02, 0.01, 0.03, 0.05, 0.85, 0.02, 0.01, 0.01]  # 8 emotions
}
```

**Stress Mapping (Same as Pipeline A):**
```python
emotion_class = yolo_output['class'][0]  # e.g., 4 (angry)

if emotion_class in [0, 1, 2]:  # Neutral, Calm, Happy
    stress_label = 0  # Not Stressed
else:  # Sad, Angry, Fearful, Disgust, Surprised
    stress_label = 1  # Stressed
```

**Advantages over Pipeline A (dlib):**
✅ **End-to-End:** No manual feature engineering  
✅ **Robust:** Handles pose variations, occlusions better  
✅ **Fast:** Real-time inference (100+ FPS on GPU)  
✅ **Scalable:** Can be retrained on larger datasets easily  

**Trade-offs:**
⚠️ **Black Box:** Less interpretable than handcrafted features  
⚠️ **Data Hungry:** Requires large labeled dataset for training  
⚠️ **Compute:** Needs GPU for efficient training  

---

### 2D MFCC SPECTROGRAMS FOR AUDIO

**Why 2D Spectrograms?**

**Pipeline A (1D Features):**
- Aggregated statistics (mean, std) over time
- Loses temporal dynamics (pitch contours, rhythm)
- Compact (17 features) but limited expressiveness

**Pipeline B (2D Spectrograms):**
- Preserves time-frequency structure
- Treats audio as "image" for CNN processing
- Captures temporal patterns (e.g., rising pitch, pauses)

---

### SPECTROGRAM CONSTRUCTION

**Step-by-Step Process:**

**1. MFCC Extraction (librosa):**
```python
import librosa

# Load audio
y, sr = librosa.load(audio_file, sr=22050)

# Compute MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# Output shape: (40, time_frames)
```

**2. Delta Features (Temporal Derivatives):**
```python
# First-order delta (velocity)
delta = librosa.feature.delta(mfcc)

# Second-order delta (acceleration)
delta2 = librosa.feature.delta(mfcc, order=2)

# Stack all features
stacked = np.vstack([mfcc, delta, delta2])
# Output shape: (120, time_frames)  [40×3 = 120 channels]
```

**3. Padding/Truncation to Fixed Length:**
```python
MAX_PAD_LEN = 174  # Fixed time dimension

if stacked.shape[1] < MAX_PAD_LEN:
    # Pad with zeros
    pad_width = MAX_PAD_LEN - stacked.shape[1]
    stacked = np.pad(stacked, pad_width=((0,0),(0,pad_width)), mode='constant')
else:
    # Truncate
    stacked = stacked[:, :MAX_PAD_LEN]

# Final shape: (120, 174)
```

**4. Reshape for CNN:**
```python
# Add channel dimension (grayscale image)
spectrogram = stacked[..., np.newaxis]
# Final shape: (120, 174, 1)
```

**Interpretation:**
- **Rows (120):** Frequency bins (40 MFCCs + 40 deltas + 40 delta-deltas)
- **Columns (174):** Time frames (~3-5 seconds of audio at 22 kHz)
- **Channels (1):** Grayscale (single channel, like a black-and-white image)

**Visual Analogy:**
- Imagine a heatmap where:
  - **X-axis:** Time progression (left to right)
  - **Y-axis:** Frequency components (low to high)
  - **Color intensity:** MFCC coefficient magnitude

---

## SLIDE 11: METHODOLOGY IV - PIPELINE B (END-TO-END)
### Part 2: Model Architecture

### 2D-CNN FOR AUDIO SPECTROGRAMS

**Architecture Diagram:**

```
INPUT LAYER
    ↓
┌─────────────────────────────────────┐
│  Input: (120, 174, 1)               │  ← 2D Spectrogram (frequency × time × channel)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Conv2D Layer 1                     │
│  Filters: 32                        │
│  Kernel: (3, 3)                     │
│  Activation: ReLU                   │
│  Padding: Same                      │
│  ↓                                   │
│  Output: (120, 174, 32)             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  BatchNormalization                 │
│  Purpose: Stabilize training        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  MaxPooling2D                       │
│  Pool Size: (2, 2)                  │
│  ↓                                   │
│  Output: (60, 87, 32)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Conv2D Layer 2                     │
│  Filters: 64                        │
│  Kernel: (3, 3)                     │
│  Activation: ReLU                   │
│  Padding: Same                      │
│  ↓                                   │
│  Output: (60, 87, 64)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  BatchNormalization                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  MaxPooling2D                       │
│  Pool Size: (2, 2)                  │
│  ↓                                   │
│  Output: (30, 43, 64)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Conv2D Layer 3                     │
│  Filters: 128                       │
│  Kernel: (3, 3)                     │
│  Activation: ReLU                   │
│  Padding: Same                      │
│  ↓                                   │
│  Output: (30, 43, 128)              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  BatchNormalization                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  MaxPooling2D                       │
│  Pool Size: (2, 2)                  │
│  ↓                                   │
│  Output: (15, 21, 128)              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  GlobalAveragePooling2D             │
│  Purpose: Reduce to 1D vector      │
│  ↓                                   │
│  Output: (128,)                     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Dense Layer (Hidden)               │
│  Units: 128                         │
│  Activation: ReLU                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Dropout Layer                      │
│  Rate: 0.4 (40% dropout)            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Dense Layer (Output)               │
│  Units: 8 (for 8 emotions)          │
│  Activation: Softmax                │
│  ↓                                   │
│  Output: [p1, p2, ..., p8]          │
└─────────────────────────────────────┘
```

**Model Parameters:**
- **Total Parameters:** 111,112
- **Trainable Parameters:** 110,664
- **Non-trainable Parameters:** 448 (BatchNorm statistics)
- **Model Size:** 434 KB

**Training Configuration:**
```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Multi-class classification
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=40,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=6, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

**Why This Architecture?**

**1. Convolutional Layers:**
- **3×3 Kernels:** Capture local time-frequency patterns (e.g., formants, harmonics)
- **Increasing Filters (32→64→128):** Learn hierarchical features (edges → textures → objects)
- **Padding='same':** Preserve spatial dimensions

**2. Batch Normalization:**
- Normalizes activations to mean=0, std=1
- Speeds up training, reduces sensitivity to initialization
- Acts as mild regularization

**3. Max Pooling:**
- Downsamples by factor of 2 (reduces computation)
- Provides translation invariance (small shifts don't affect output)
- Extracts dominant features

**4. Global Average Pooling:**
- Alternative to Flatten + Dense
- Reduces overfitting (fewer parameters)
- Averages each feature map to single value

**5. Dropout (0.4):**
- Randomly drops 40% of neurons during training
- Prevents co-adaptation (forces redundancy)
- Improves generalization

---

### TRAINING RESULTS (2D-CNN Audio Model)

**Performance Metrics:**

**Test Accuracy: 96.53%** ⭐

**Classification Report:**
```
              precision    recall  f1-score   support

       angry       0.97      1.00      0.99        76
        calm       0.99      1.00      0.99        77
     disgust       0.99      0.95      0.97        77
     fearful       0.97      0.91      0.94        77
       happy       0.91      1.00      0.95        77
     neutral       1.00      0.95      0.97        38
         sad       0.93      0.91      0.92        77
   surprised       0.99      1.00      0.99        77

    accuracy                           0.97       576
   macro avg       0.97      0.96      0.97       576
weighted avg       0.97      0.97      0.97       576
```

**Key Observations:**
- **Excellent Overall Performance:** 96.53% accuracy on 8-class emotion recognition
- **Balanced Precision/Recall:** No single class dominates
- **Best Performance:** Angry, Calm, Surprised (F1 ≥ 0.99)
- **Challenging Classes:** Fearful (F1 = 0.94), Sad (F1 = 0.92)
  - Likely due to acoustic similarity (both low-arousal, negative valence)

**Confusion Matrix Insights:**
- **Angry:** 100% recall (all angry samples correctly identified)
- **Happy:** 100% recall (all happy samples correctly identified)
- **Fearful vs. Sad:** Main confusion (fearful sometimes classified as sad)
- **Neutral:** 95% recall (some neutral misclassified as calm - understandable overlap)

---

### YOLOV11N FACIAL MODEL (Expected Performance)

**Note:** Detailed results depend on custom facial emotion dataset used for fine-tuning.

**Typical YOLOv11n Performance on Facial Emotion:**
- **mAP@0.5:** 85-92% (mean Average Precision at IoU threshold 0.5)
- **mAP@0.75:** 70-80% (stricter IoU threshold)
- **Inference Speed:** 100+ FPS on GPU, 20-30 FPS on CPU
- **Detection Accuracy:** 95%+ (face detection rate)
- **Classification Accuracy:** 80-90% (emotion classification given detected face)

**Advantages:**
✅ **Real-time:** Suitable for live video streams  
✅ **Robust:** Handles multiple faces, occlusions, pose variations  
✅ **Scalable:** Can detect + classify in single pass  

**Limitations:**
⚠️ **Dataset Dependent:** Performance varies with training data quality  
⚠️ **Black Box:** Difficult to interpret learned features  
⚠️ **Compute:** Requires GPU for real-time performance  

---

### PIPELINE B SUMMARY

**Facial Stream (YOLOv11n):**
- **Input:** Raw video frames (1920×1080)
- **Processing:** Resize → YOLO detection → Emotion classification
- **Output:** Emotion class (0-7) → Stress label (0 or 1)
- **Model Size:** 5.6 MB
- **Inference:** Real-time (100+ FPS on GPU)

**Audio Stream (2D-CNN):**
- **Input:** Raw audio waveform
- **Processing:** MFCC spectrogram (120×174×1) → 2D-CNN
- **Output:** 8-class probabilities → Stress label
- **Model Size:** 434 KB
- **Accuracy:** 96.53% on 8-class emotion recognition

**Key Differences from Pipeline A:**
- **No Manual Feature Engineering:** Models learn features automatically
- **Higher Capacity:** More parameters (111K vs. 67K for BiLSTM)
- **Better Accuracy (Audio):** 96.53% vs. ~70% for BiLSTM
- **Less Interpretable:** Cannot easily visualize learned features
- **More Data Hungry:** Requires larger datasets for optimal performance

---

