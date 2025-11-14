# MULTIMODAL STRESS DETECTION SYSTEM
## Presentation Content - Part 3: Pipeline A (Engineered Features)

---

## SLIDE 8: METHODOLOGY III - PIPELINE A (ENGINEERED)
### Part 1: Feature Extraction

### FACIAL FEATURE EXTRACTION (dlib-based)

**Tool:** dlib 68-point facial landmark detector

**Processing Pipeline:**

```
Video Frame (1920×1080)
    ↓
Resize to 480px width (speed optimization)
    ↓
dlib Frontal Face Detector
    ↓
68-Point Landmark Detection
    ↓
Feature Engineering (4 features per frame)
    ↓
Sequence Construction (150 frames)
    ↓
Output: (150, 4) array per video
```

**Extracted Features (Per Frame):**

**1. EAR - Eye Aspect Ratio**
```
Formula: EAR = (||p1 - p5|| + ||p2 - p4||) / (2 × ||p0 - p3||)

Where:
- p0, p3 = Outer and inner eye corners (horizontal)
- p1, p2, p4, p5 = Vertical eye landmarks

Purpose: Measures eye openness
- High EAR → Wide open eyes (alertness, surprise, fear)
- Low EAR → Closed/squinting eyes (fatigue, disgust, sadness)
- Averaged across both eyes for robustness
```

**2. MAR - Mouth Aspect Ratio**
```
Formula: MAR = (||q13 - q19|| + ||q14 - q18|| + ||q15 - q17||) / (2 × ||q12 - q16||)

Where:
- q12, q16 = Left and right mouth corners (horizontal)
- q13-q19, q14-q18, q15-q17 = Vertical mouth landmarks

Purpose: Measures mouth openness
- High MAR → Open mouth (surprise, fear, yelling in anger)
- Low MAR → Closed mouth (calm, neutral, sadness)
```

**3. Eyebrow Distance**
```
Formula: Eyebrow_Dist = mean_eyebrow_y - mean_eye_y

Where:
- mean_eyebrow_y = Average y-coordinate of eyebrow landmarks (17-26)
- mean_eye_y = Average y-coordinate of eye landmarks (36-47)

Purpose: Measures eyebrow elevation
- Positive (large) → Raised eyebrows (surprise, fear)
- Negative (small) → Lowered/furrowed brows (anger, concentration, stress)
```

**4. Mouth Corner Elevation**
```
Formula: Mouth_Elev = mean_inner_lip_y - mean_mouth_corner_y

Where:
- mean_inner_lip_y = Average y-coordinate of inner lip landmarks
- mean_mouth_corner_y = Average y-coordinate of mouth corners (48, 54)

Purpose: Measures smile/frown
- Positive → Corners raised (happiness, smile)
- Negative → Corners lowered (sadness, frown, disgust)
```

**Processing Configuration:**
- **Frame Skip:** 5 (process every 5th frame for efficiency)
- **Resize Width:** 480px (maintains aspect ratio, speeds up detection)
- **Sequence Length:** 150 frames (padded or truncated)
- **Padding Strategy:** Zero-padding for videos with <150 frames
- **Missing Face Handling:** Zero-vector if face not detected in frame

**Output:**
- **File:** `X_facial.npy`
- **Shape:** (2880, 150, 4)
- **Size:** 13.8 MB
- **Interpretation:** 2,880 videos × 150 time steps × 4 features

---

### AUDIO FEATURE EXTRACTION (librosa 1D)

**Tool:** librosa audio analysis library

**Processing Pipeline:**

```
MP4 Video File
    ↓
Extract audio with MoviePy
    ↓
Resample to 16 kHz (from 48 kHz)
    ↓
Load with librosa
    ↓
Feature Extraction (1D aggregated features)
    ↓
Concatenate into single vector
    ↓
Output: (17,) array per video
```

**Extracted Features (Per Video):**

**1. MFCCs - Mel-Frequency Cepstral Coefficients**
```
Configuration:
- n_mfcc = 13 (standard for speech recognition)
- Aggregation: Mean across time

Output: 13 features (MFCC_1 to MFCC_13)

Purpose: Captures timbre and spectral envelope
- Represents vocal tract shape and phonetic content
- Lower coefficients (1-3) → Broad spectral shape
- Higher coefficients (4-13) → Fine spectral details
- Sensitive to voice quality changes under stress
```

**2. Pitch (Fundamental Frequency)**
```
Extraction Method:
- librosa.piptrack() for pitch tracking
- Select bins with magnitude > median
- Robust mean of energetic pitch bins

Output: 1 feature (mean pitch in Hz)

Purpose: Vocal pitch variation
- High pitch → Excitement, fear, anger (high arousal)
- Low pitch → Sadness, calmness (low arousal)
- Pitch variability indicates emotional intensity
```

**3. ZCR - Zero-Crossing Rate**
```
Formula: ZCR = (1/T) × Σ |sign(x[t]) - sign(x[t-1])|

Output: 1 feature (mean ZCR)

Purpose: Measures signal noisiness
- High ZCR → Unvoiced sounds (fricatives, whispers, breathy speech)
- Low ZCR → Voiced sounds (vowels, steady tones)
- Increases with stress-induced vocal tension
```

**4. Spectral Contrast**
```
Configuration:
- librosa.feature.spectral_contrast()
- 7 frequency bands (default)
- Aggregation: Mean across time

Output: 2 features (mean spectral contrast, std)

Purpose: Difference between peaks and valleys in spectrum
- High contrast → Clear, articulated speech
- Low contrast → Mumbled, monotone speech
- Changes with emotional arousal and stress
```

**Total Feature Vector:**
- **MFCCs:** 13 features
- **Pitch:** 1 feature
- **ZCR:** 1 feature
- **Spectral Contrast:** 2 features
- **TOTAL:** 17 features per video

**Output:**
- **File:** `X_audio.npy`
- **Shape:** (2880, 17)
- **Size:** 507 KB
- **Interpretation:** 2,880 videos × 17 aggregated audio features

---

### FEATURE EXTRACTION LOGS

**Detailed Logging for Reproducibility:**

**audio_extraction.log (265 KB):**
- Processing time per video
- Feature extraction success/failure
- Missing or corrupted files
- Final feature statistics (mean, std per feature)

**facial_extraction.log (271 KB):**
- Face detection success rate
- Landmark detection failures
- Frame-by-frame processing details
- Sequence padding/truncation counts

**Key Statistics from Logs:**
- **Facial Detection Success Rate:** ~98% (face detected in at least 1 frame)
- **Average Processing Time:** 
  - Facial: ~2-3 seconds per video
  - Audio: ~1-2 seconds per video
- **Total Extraction Time:** ~2-3 hours for full dataset (2,880 videos)

---

## SLIDE 9: METHODOLOGY III - PIPELINE A (ENGINEERED)
### Part 2: Model Architecture

### BILSTM MODEL (Primary Architecture)

**Architecture Diagram:**

```
INPUT LAYER
    ↓
┌─────────────────────────────────────┐
│  Facial Input: (150, 4)             │  ← Sequence of 150 frames, 4 features each
│  Audio Input: (1, 17)               │  ← Reshaped to (1 timestep, 17 features)
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Bidirectional LSTM Layer 1         │
│  Units: 64 (32 forward + 32 back)   │
│  return_sequences=True              │
│  ↓                                   │
│  Output: (timesteps, 128)           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Bidirectional LSTM Layer 2         │
│  Units: 32 (16 forward + 16 back)   │
│  return_sequences=False             │
│  ↓                                   │
│  Output: (64,)                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Dropout Layer                      │
│  Rate: 0.5 (50% dropout)            │
│  Purpose: Regularization            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Dense Layer (Hidden)               │
│  Units: 32                          │
│  Activation: ReLU                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Dense Layer (Output)               │
│  Units: 1                           │
│  Activation: Sigmoid                │
│  ↓                                   │
│  Output: Probability [0, 1]         │
└─────────────────────────────────────┘
```

**Why BiLSTM?**

1. **Bidirectional Processing:**
   - Forward LSTM: Captures past context (frame t-1, t-2, ...)
   - Backward LSTM: Captures future context (frame t+1, t+2, ...)
   - Combined: Full temporal context for each timestep

2. **Sequence Learning:**
   - Facial features evolve over time (e.g., gradual smile, progressive eye closure)
   - Audio features have temporal dependencies (pitch contours, rhythm)
   - LSTM cells maintain long-term memory via gates (forget, input, output)

3. **Handles Variable-Length Sequences:**
   - Padding/truncation to fixed length (150 frames)
   - LSTM naturally processes sequences of any length

**Model Parameters:**

**Facial BiLSTM:**
- **Input Shape:** (150, 4)
- **Total Parameters:** ~67,000
- **Trainable Parameters:** ~67,000
- **Model Size:** ~260 KB

**Audio BiLSTM:**
- **Input Shape:** (1, 17) [reshaped from (17,)]
- **Total Parameters:** ~45,000
- **Trainable Parameters:** ~45,000
- **Model Size:** ~180 KB

**Training Configuration:**
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy
- **Epochs:** 50
- **Batch Size:** 32
- **Validation Split:** 10% of training data (for early stopping)

---

### ALTERNATIVE ARCHITECTURES (Comparative Study)

**1. BiGRU (Bidirectional Gated Recurrent Unit)**

```
Architecture:
- Bidirectional GRU(64, return_sequences=True)
- Bidirectional GRU(32)
- Dropout(0.5)
- Dense(32, relu)
- Dense(1, sigmoid)

Differences from BiLSTM:
- GRU has 2 gates (reset, update) vs LSTM's 3 (forget, input, output)
- Fewer parameters → Faster training
- Comparable performance on many tasks

Results:
- Facial BiGRU Accuracy: ~68-72%
- Audio BiGRU Accuracy: ~67-71%
- Training Time: ~15% faster than BiLSTM
```

**2. 1D-CNN (1-Dimensional Convolutional Neural Network)**

```
Architecture:
- Conv1D(64, kernel_size=3, relu)
- Conv1D(64, kernel_size=3, relu)
- GlobalMaxPooling1D()
- Dropout(0.5)
- Dense(32, relu)
- Dense(1, sigmoid)

Differences from BiLSTM:
- Convolutional filters learn local patterns
- Max pooling extracts most salient features
- No recurrent connections → Faster inference

Results:
- Facial 1D-CNN Accuracy: ~65-69%
- Audio 1D-CNN Accuracy: ~64-68%
- Inference Time: ~3x faster than BiLSTM
```

**3. DNN (Dense Neural Network / MLP)**

```
Architecture:
- Dense(64, relu, input_shape=(17,))  ← For audio only
- Dropout(0.5)
- Dense(32, relu)
- Dense(1, sigmoid)

Differences from BiLSTM:
- No sequence modeling (treats features as independent)
- Simplest architecture
- Suitable for aggregated features (audio 1D)

Results:
- Audio DNN Accuracy: ~66-70%
- Not suitable for facial sequences (would require flattening)
- Fastest training and inference
```

---

### MODEL COMPARISON SUMMARY (Pipeline A)

| Model | Facial Accuracy | Audio Accuracy | Training Time | Inference Speed | Parameters |
|-------|----------------|----------------|---------------|-----------------|------------|
| **BiLSTM** | **70.6%** | **69.6%** | Baseline (100%) | Baseline (100%) | ~67K (facial) |
| **BiGRU** | 68-72% | 67-71% | 85% of BiLSTM | 110% of BiLSTM | ~55K (facial) |
| **1D-CNN** | 65-69% | 64-68% | 60% of BiLSTM | 300% of BiLSTM | ~45K (facial) |
| **DNN** | N/A | 66-70% | 40% of BiLSTM | 500% of BiLSTM | ~3K (audio) |

**Key Insights:**
- **BiLSTM** achieves best accuracy due to temporal modeling
- **BiGRU** offers good accuracy-speed trade-off
- **1D-CNN** fastest inference, suitable for real-time applications
- **DNN** simplest baseline, good for audio-only quick prototyping

---

### TRAINING PROCESS

**Data Preparation:**
```python
# Stratified split
X_facial_train, X_facial_test, Y_train, Y_test = train_test_split(
    X_facial, Y, test_size=0.2, random_state=42, stratify=Y
)

# Audio reshaping for LSTM
X_audio_train = np.expand_dims(X_audio_train, axis=1)  # (N, 17) → (N, 1, 17)
```

**Training Loop:**
```python
history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)
```

**Regularization Techniques:**
1. **Dropout (0.5):** Prevents overfitting by randomly dropping 50% of neurons
2. **Validation Monitoring:** Track val_loss to detect overfitting
3. **Stratified Split:** Ensures balanced class distribution in train/test

**Computational Requirements:**
- **Hardware:** CPU-friendly (no GPU required, though GPU accelerates training)
- **RAM:** ~4 GB for full dataset in memory
- **Training Time:** 
  - Facial BiLSTM: ~10-15 minutes (50 epochs)
  - Audio BiLSTM: ~5-10 minutes (50 epochs)
- **Total Training Time:** ~20-25 minutes for both modalities

---

### INTERPRETABILITY ADVANTAGES

**Feature-Level Analysis:**

**Facial Features:**
- **EAR:** Can plot EAR over time to see blink patterns (stress → reduced blinking)
- **MAR:** Visualize mouth openness trajectory (yelling in anger, gasping in fear)
- **Eyebrow Distance:** Track eyebrow raising (surprise) or furrowing (anger)
- **Mouth Corner:** Quantify smile/frown intensity

**Audio Features:**
- **MFCCs 1-3:** Dominant spectral shape (voice quality)
- **Pitch:** Track pitch contour (rising pitch in questions, falling in statements)
- **ZCR:** Identify breathy/whispered segments (fear, sadness)
- **Spectral Contrast:** Measure articulation clarity (stress → reduced contrast)

**Model Explainability:**
- **LSTM Attention:** Can visualize which timesteps (frames) contribute most to prediction
- **Feature Importance:** Analyze which of the 4 facial or 17 audio features are most predictive
- **Error Analysis:** Identify which emotions are confused (e.g., fear vs. surprise)

**Clinical/Research Value:**
- Researchers can validate features against psychological theory
- Clinicians can understand WHY a prediction was made
- Enables hypothesis testing (e.g., "Does eyebrow distance correlate with stress?")

---

