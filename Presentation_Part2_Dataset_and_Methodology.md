# MULTIMODAL STRESS DETECTION SYSTEM
## Presentation Content - Part 2: Dataset & Methodology

---

## SLIDE 5: DATASET DEEP DIVE - RAVDESS

### Why RAVDESS?

**RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song**

**Selection Rationale:**
1. **Multimodal by Design:** Synchronized audio and video in every sample - perfect for our fusion approach
2. **Controlled Quality:** Professional recording studio with consistent lighting, camera angles, and audio equipment
3. **Diverse Actors:** 24 professional actors (12 male, 12 female) ensuring gender balance
4. **Rich Emotional Range:** 8 distinct emotions with varying intensity levels
5. **Publicly Available:** Enables reproducibility and comparison with other research
6. **Sufficient Scale:** 2,880 samples provide adequate data for deep learning while remaining manageable

### Dataset Structure & Organization

**File Hierarchy:**
```
Dataset/
├── Video_Speech_Actor_01/
│   └── Actor_01/
│       ├── 01-01-01-01-01-01-01.mp4
│       ├── 01-01-02-01-01-01-01.mp4
│       └── ... (120 files per actor)
├── Video_Speech_Actor_02/
│   └── Actor_02/
│       └── ... (120 files)
...
└── Video_Speech_Actor_24/
    └── Actor_24/
        └── ... (120 files)
```

**Total Dataset Size:**
- **2,880 video files** (24 actors × 120 samples each)
- **File Format:** MP4 (H.264 video codec, AAC audio codec)
- **Average File Size:** 4-5 MB per video
- **Total Dataset Size:** ~12 GB

### Filename Encoding Schema

**Format:** `Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor.mp4`

**Example:** `01-01-03-02-01-02-01.mp4`

**Breakdown:**
- **Position 1 (Modality):** 
  - `01` = Audio-Video (full recording)
  - `02` = Video-only
  - `03` = Audio-only

- **Position 2 (Vocal Channel):**
  - `01` = Speech
  - `02` = Song

- **Position 3 (Emotion):** ⭐ **CRITICAL FOR OUR LABELING**
  - `01` = Neutral
  - `02` = Calm
  - `03` = Happy
  - `04` = Sad
  - `05` = Angry
  - `06` = Fearful
  - `07` = Disgust
  - `08` = Surprised

- **Position 4 (Intensity):**
  - `01` = Normal intensity
  - `02` = Strong intensity

- **Position 5 (Statement):**
  - `01` = "Kids are talking by the door"
  - `02` = "Dogs are sitting by the door"

- **Position 6 (Repetition):**
  - `01` = 1st repetition
  - `02` = 2nd repetition

- **Position 7 (Actor):**
  - `01` to `24` = Actor ID

### Actor Demographics

**Gender Distribution:**
- **Male Actors:** 01-24 (odd numbers) = 12 actors
- **Female Actors:** 01-24 (even numbers) = 12 actors
- **Perfect 50-50 balance** for gender-neutral model training

**Age Range:**
- Young adults to middle-aged (20-45 years)
- Professional actors with emotion expression training

### Emotion Distribution (Original)

| Emotion | Code | Count per Actor | Total Samples |
|---------|------|-----------------|---------------|
| Neutral | 01 | 15 | 360 |
| Calm | 02 | 15 | 360 |
| Happy | 03 | 15 | 360 |
| Sad | 04 | 15 | 360 |
| Angry | 05 | 15 | 360 |
| Fearful | 06 | 15 | 360 |
| Disgust | 07 | 15 | 360 |
| Surprised | 08 | 15 | 360 |
| **TOTAL** | - | **120** | **2,880** |

### Technical Specifications

**Video Properties:**
- **Resolution:** 1920×1080 (Full HD)
- **Frame Rate:** 30 fps
- **Duration:** 3-5 seconds per clip
- **Codec:** H.264/AVC

**Audio Properties:**
- **Sample Rate:** 48 kHz (original), downsampled to 16 kHz for processing
- **Bit Depth:** 16-bit
- **Channels:** Mono (single speaker)
- **Codec:** AAC

### Dataset Advantages for Our Project

✅ **Synchronized Modalities:** No need for manual audio-video alignment  
✅ **Consistent Quality:** Minimal preprocessing required  
✅ **Balanced Classes:** Equal representation prevents bias  
✅ **Professional Acting:** Clear, exaggerated expressions ideal for training  
✅ **Reproducible:** Other researchers can validate our results  
✅ **Adequate Scale:** 2,880 samples sufficient for BiLSTM and CNN training  

### Dataset Limitations (Acknowledged)

⚠️ **Acted Emotions:** May not fully represent spontaneous real-world stress  
⚠️ **Controlled Environment:** Studio setting lacks real-world noise/lighting variations  
⚠️ **Limited Age Range:** Primarily young adults, may not generalize to children or elderly  
⚠️ **Cultural Homogeneity:** North American actors, may not capture cross-cultural expressions  

---

## SLIDE 6: METHODOLOGY I - DATA LABELING

### Our "Proxy Label" for Stress - Critical Design Decision

**The Challenge:**
- RAVDESS provides **emotion labels** (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- Our goal is **stress detection** (binary: Stressed vs. Not Stressed)
- **No direct stress labels exist in the dataset**

### Our Labeling Strategy: Emotion-to-Stress Mapping

**Mapping Rule (Implemented in both extractors):**

```python
# From audio_extractor.py and facial_extractor.py
def get_stress_label(emotion_code):
    """
    Map RAVDESS emotion codes to binary stress labels.
    
    Emotion codes from filename position 3:
    - 1, 2, 3 → 0 (Not Stressed)
    - 4, 5, 6, 7, 8 → 1 (Stressed)
    """
    emotion_code = int(filename.split('-')[2])
    
    if emotion_code in [1, 2, 3]:  # Neutral, Calm, Happy
        return 0  # Not Stressed
    else:  # Sad, Angry, Fearful, Disgust, Surprised
        return 1  # Stressed
```

### Theoretical Justification

**Not Stressed (Label 0):**
- **Neutral (01):** Baseline emotional state, no arousal
- **Calm (02):** Relaxed, low arousal, positive valence
- **Happy (03):** Positive emotion, associated with low stress

**Stressed (Label 1):**
- **Sad (04):** Negative valence, associated with chronic stress
- **Angry (05):** High arousal, fight response, acute stress
- **Fearful (06):** High arousal, flight response, acute stress
- **Disgust (07):** Negative valence, aversive stress
- **Surprised (08):** High arousal, uncertainty-induced stress

**Psychological Basis:**
- Aligns with **Russell's Circumplex Model** of affect (arousal × valence)
- Stressed emotions cluster in high-arousal or negative-valence quadrants
- Supported by **Lazarus's Transactional Model** of stress and coping

### Resulting Label Distribution

**After Mapping:**

| Stress Label | Emotions Included | Original Count | Final Count | Percentage |
|--------------|-------------------|----------------|-------------|------------|
| **0 (Not Stressed)** | Neutral, Calm, Happy | 360 + 360 + 360 | **1,080** | **37.5%** |
| **1 (Stressed)** | Sad, Angry, Fearful, Disgust, Surprised | 360 × 5 | **1,800** | **62.5%** |
| **TOTAL** | All 8 emotions | 2,880 | **2,880** | **100%** |

**Class Imbalance:**
- **Imbalance Ratio:** 1.67:1 (Stressed:Not Stressed)
- **Mitigation Strategy:** Stratified train-test split ensures proportional representation
- **Impact:** Models may have slight bias toward "Stressed" class, monitored via precision-recall metrics

### Critical Acknowledgments

**⚠️ This is a PROXY LABEL, not ground truth stress:**

**Limitations:**
1. **Emotion ≠ Stress:** Not all angry or fearful expressions indicate chronic stress
2. **Context Matters:** Same emotion can be stress-related or not depending on situation
3. **Individual Differences:** Stress manifests differently across people
4. **Acted Data:** Professional actors may exaggerate expressions beyond real stress

**Why We Proceed Anyway:**
1. **Proof of Concept:** Demonstrates multimodal fusion effectiveness on related task
2. **Established Practice:** Many emotion recognition papers use similar mappings
3. **Consistent Application:** Same mapping used across all modalities ensures fair comparison
4. **Future Extensibility:** Framework can be retrained on true stress-labeled data

**Validation Approach:**
- Both `audio_extractor.py` and `facial_extractor.py` use **identical mapping**
- Assertion check in `collab.py`: `assert np.array_equal(Y_facial, Y_audio)`
- Ensures no label mismatch between modalities

### Label Files Generated

**Output Files:**
- `Y_audio_labels.npy` - Shape: (2880,) - Binary labels from audio extraction
- `Y_facial_labels.npy` - Shape: (2880,) - Binary labels from facial extraction
- **Verified Identical:** Both files contain the same label sequence

**Train-Test Split (Stratified):**
- **Training Set:** 2,304 samples (80%)
  - Not Stressed: 864 samples (37.5%)
  - Stressed: 1,440 samples (62.5%)
- **Test Set:** 576 samples (20%)
  - Not Stressed: 216 samples (37.5%)
  - Stressed: 360 samples (62.5%)

### Ethical Considerations

**Transparency:**
- We explicitly disclose this is a proxy label in all documentation
- Results should NOT be interpreted as clinical stress detection
- Suitable for research and proof-of-concept, not medical diagnosis

**Future Work:**
- Collect dataset with validated stress labels (cortisol levels, self-reported stress scales)
- Explore alternative mappings (e.g., arousal-based, valence-based)
- Investigate multi-task learning (predict both emotion and stress simultaneously)

---

## SLIDE 7: METHODOLOGY II - SYSTEM ARCHITECTURE

### High-Level System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAVDESS DATASET (2,880 MP4 Files)           │
│                  24 Actors × 120 Samples Each                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │   DATA INGESTION & LABEL EXTRACTION    │
        │  Parse filename → Emotion Code (Pos 3) │
        │  Map to Binary Stress Label (0 or 1)   │
        └────────┬───────────────────────┬────────┘
                 │                       │
    ┌────────────▼────────────┐ ┌───────▼──────────────┐
    │   FACIAL EXTRACTION     │ │   AUDIO EXTRACTION   │
    │   (facial_extractor.py) │ │  (audio_extractor.py)│
    └────────┬────────────────┘ └──────┬───────────────┘
             │                         │
             ▼                         ▼
┌────────────────────────┐  ┌─────────────────────────┐
│  PIPELINE A:           │  │  PIPELINE B:            │
│  ENGINEERED FEATURES   │  │  END-TO-END DEEP        │
├────────────────────────┤  ├─────────────────────────┤
│ FACIAL:                │  │ FACIAL:                 │
│ • dlib 68 landmarks    │  │ • YOLOv11n detection    │
│ • EAR, MAR, eyebrow    │  │ • Transfer learning     │
│ • Sequence (150, 4)    │  │ • Bounding box + class  │
│                        │  │                         │
│ AUDIO:                 │  │ AUDIO:                  │
│ • librosa 1D features  │  │ • 2D MFCC spectrograms  │
│ • MFCCs, pitch, ZCR    │  │ • Delta + Delta-Delta   │
│ • Spectral contrast    │  │ • Shape (120, 174, 1)   │
└────────┬───────────────┘  └──────┬──────────────────┘
         │                         │
         ▼                         ▼
┌────────────────────────┐  ┌─────────────────────────┐
│  X_facial.npy          │  │  (Processed on-the-fly  │
│  Shape: (2880, 150, 4) │  │   or pre-extracted)     │
│                        │  │                         │
│  X_audio.npy           │  │                         │
│  Shape: (2880, 17)     │  │                         │
└────────┬───────────────┘  └──────┬──────────────────┘
         │                         │
         ▼                         ▼
┌────────────────────────┐  ┌─────────────────────────┐
│  TRAIN-TEST SPLIT      │  │  TRAIN-TEST SPLIT       │
│  80% Train / 20% Test  │  │  80% Train / 20% Test   │
│  Stratified by label   │  │  Stratified by label    │
│  random_state=42       │  │  random_state=42        │
└────────┬───────────────┘  └──────┬──────────────────┘
         │                         │
         ▼                         ▼
┌────────────────────────┐  ┌─────────────────────────┐
│  MODEL TRAINING        │  │  MODEL TRAINING         │
│  • BiLSTM (facial)     │  │  • YOLOv11n (facial)    │
│  • BiLSTM (audio)      │  │  • 2D-CNN (audio)       │
│  • BiGRU, 1D-CNN, DNN  │  │  • Transfer learning    │
│  • 50 epochs, batch=32 │  │  • 80 epochs (YOLO)     │
└────────┬───────────────┘  └──────┬──────────────────┘
         │                         │
         ▼                         ▼
┌────────────────────────┐  ┌─────────────────────────┐
│  UNIMODAL PREDICTIONS  │  │  UNIMODAL PREDICTIONS   │
│  P_facial, P_audio     │  │  P_facial, P_audio      │
└────────┬───────────────┘  └──────┬──────────────────┘
         │                         │
         └────────┬────────────────┘
                  ▼
         ┌────────────────────┐
         │  LATE FUSION       │
         │  P_fused = (P_f +  │
         │            P_a) / 2│
         └────────┬───────────┘
                  ▼
         ┌────────────────────┐
         │  FINAL PREDICTION  │
         │  Stressed / Not    │
         └────────┬───────────┘
                  ▼
         ┌────────────────────┐
         │  EVALUATION        │
         │  • Accuracy        │
         │  • Precision/Recall│
         │  • F1-Score        │
         │  • Confusion Matrix│
         └────────────────────┘
```

### Key System Components

**1. Data Layer:**
- **Input:** RAVDESS MP4 files organized by actor
- **Parsing:** Filename-based emotion code extraction
- **Labeling:** Binary stress mapping (emotion codes 1-3 → 0, 4-8 → 1)

**2. Feature Extraction Layer:**
- **Facial Pipeline:** `facial_extractor.py` (dlib or YOLOv11)
- **Audio Pipeline:** `audio_extractor.py` (librosa or 2D spectrograms)
- **Output:** Serialized `.npy` arrays for fast loading

**3. Model Layer:**
- **Pipeline A:** BiLSTM/BiGRU/1D-CNN/DNN on engineered features
- **Pipeline B:** YOLOv11n (facial) + 2D-CNN (audio) end-to-end
- **Training:** TensorFlow/Keras with Adam optimizer

**4. Fusion Layer:**
- **Strategy:** Late fusion (probability averaging)
- **Input:** Softmax probabilities from each modality
- **Output:** Combined prediction with higher confidence

**5. Evaluation Layer:**
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Visualization:** Confusion matrices, classification reports
- **Comparison:** Unimodal vs. multimodal performance

### Data Flow Summary

1. **Ingestion:** 2,880 videos → Filename parsing → Binary labels
2. **Extraction:** Videos → Facial features (150×4) + Audio features (17D or spectrograms)
3. **Storage:** Features saved as `.npy` files (X_facial, X_audio, Y_labels)
4. **Splitting:** Stratified 80-20 split (2,304 train, 576 test)
5. **Training:** Separate models for facial and audio modalities
6. **Fusion:** Average probabilities from both models
7. **Evaluation:** Compare fused vs. unimodal performance

---

