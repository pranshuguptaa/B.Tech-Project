import os
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
import logging
from tqdm import tqdm
import glob

# ==================================
# 1. CONFIGURATION
# ==================================
CONFIG = {
    "data_path": "/Users/pranshugupta/Desktop/B.Tech Project/Sample Data/",
    "temp_audio_path": "temp_audio.wav",
    "output_features": "X_audio.npy",
    "output_labels": "Y_audio_labels.npy",
    "log_file": "audio_extraction.log",
    "n_mfcc": 13
}

# ==================================
# 2. LOGGING SETUP
# ==================================
if os.path.exists(CONFIG["log_file"]):
    os.remove(CONFIG["log_file"])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)

# ==================================
# 3. HELPER FUNCTIONS
# ==================================
def get_label_from_filename(filename):
    """Parses filename to get the stress label (0 or 1)."""
    try:
        emotion_code = int(os.path.basename(filename).split('-')[2])
        return 0 if emotion_code in [1, 2, 3] else 1
    except (IndexError, ValueError):
        logging.warning(f"Could not parse emotion from filename: {filename}. Skipping label.")
        return -1

def get_audio_features(y, sr):
    """Extracts all audio features from a loaded audio signal."""
    # MFCCs
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG["n_mfcc"]).T, axis=0)
    
    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[magnitudes > np.median(magnitudes)]) if np.any(magnitudes > 0) else 0.0
    pitch = 0.0 if np.isnan(pitch) else pitch

    # Zero-Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

    # Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    
    # Combine all features into a single vector
    return np.hstack([mfccs, pitch, zcr, contrast])

# ==================================
# 4. MAIN PROCESSING LOGIC
# ==================================
def process_audio_features():
    logging.info("--- Starting Comprehensive Audio Feature Extraction ---")
    
    all_features = []
    all_labels = []

    search_pattern = os.path.join(CONFIG["data_path"], '**', '*.mp4')
    video_files = sorted(glob.glob(search_pattern, recursive=True))

    if not video_files:
        logging.error(f"No video files found at the specified path: {CONFIG['data_path']}")
        return
    logging.info(f"Found {len(video_files)} video files to process.")
    
    for video_path in tqdm(video_files, desc="Processing Audio"):
        filename = os.path.basename(video_path)
        try:
            # Extract audio using moviepy
            with VideoFileClip(video_path) as video_clip:
                video_clip.audio.write_audiofile(CONFIG["temp_audio_path"], logger=None, fps=16000)
            
            # Load audio and extract features
            y, sr = librosa.load(CONFIG["temp_audio_path"], sr=None)
            features = get_audio_features(y, sr)
            
            label = get_label_from_filename(filename)
            if label != -1:
                all_features.append(features)
                all_labels.append(label)
                logging.info(f"Successfully processed audio for {filename}")

        except Exception as e:
            logging.error(f"Could not process audio for {filename}. Error: {e}")
            continue
        finally:
            if os.path.exists(CONFIG["temp_audio_path"]):
                os.remove(CONFIG["temp_audio_path"])

    logging.info(f"--- Saving Processed Audio Data ---")
    np.save(CONFIG["output_features"], np.array(all_features))
    np.save(CONFIG["output_labels"], np.array(all_labels))
    logging.info(f"Audio feature extraction complete. Data shape: {np.array(all_features).shape}")

# ==================================
# 5. SCRIPT EXECUTION
# ==================================
if __name__ == "__main__":
    process_audio_features()