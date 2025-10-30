import os
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import logging
from tqdm import tqdm
import glob

# ==================================
# 1. CONFIGURATION
# ==================================
CONFIG = {
    # Update this to the top-level folder containing all your actor folders
    "data_path": "/Users/pranshugupta/Desktop/B.Tech Project/Sample Data/",
    "predictor_path": "/Users/pranshugupta/Desktop/B.Tech Project/shape_predictor_68_face_landmarks.dat",
    "output_features": "X_facial.npy",
    "output_labels": "Y_facial_labels.npy",
    "log_file": "facial_extraction.log",
    "sequence_length": 150,
    "num_features": 4  # EAR, MAR, Eyebrow Dist, Mouth Corner Elev
}

# ==================================
# 2. LOGGING SETUP
# ==================================
# Overwrite the log file each time the script is run
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
        return -1 # Return an invalid label

def calculate_ear(eye):
    """Calculates Eye Aspect Ratio."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def calculate_mar(mouth):
    """Calculates Mouth Aspect Ratio."""
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16]) # Horizontal distance
    return (A + B + C) / (2.0 * D) if D > 0 else 0.0

def get_feature_vector(shape):
    """Extracts all facial features from a set of landmarks."""
    ear = (calculate_ear(shape[42:48]) + calculate_ear(shape[36:42])) / 2.0
    mar = calculate_mar(shape[48:68])
    eyebrow_dist = (np.mean(shape[22:27, 1]) - np.mean(shape[42:48, 1])) + \
                   (np.mean(shape[17:22, 1]) - np.mean(shape[36:42, 1])) / 2
    corner_elev = (np.mean([shape[62, 1], shape[66, 1]]) - np.mean([shape[54, 1], shape[48, 1]]))
    return [ear, mar, eyebrow_dist, corner_elev]

# ==================================
# 4. MAIN PROCESSING LOGIC
# ==================================
def process_facial_features():
    logging.info("--- Starting Comprehensive Facial Feature Extraction ---")
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(CONFIG["predictor_path"])

    all_features = []
    all_labels = []

    search_pattern = os.path.join(CONFIG["data_path"], '**', '*.mp4')
    video_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not video_files:
        logging.error(f"No video files found at the specified path: {CONFIG['data_path']}")
        return
    logging.info(f"Found {len(video_files)} video files to process.")
    
    for video_path in tqdm(video_files, desc="Processing Faces"):
        filename = os.path.basename(video_path)
        try:
            cap = cv2.VideoCapture(video_path)
            frame_features = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                
                if len(faces) > 0:
                    shape = np.array([(predictor(gray, faces[0]).part(i).x, predictor(gray, faces[0]).part(i).y) for i in range(68)])
                    features = get_feature_vector(shape)
                    frame_features.append(features)
                else:
                    frame_features.append([0.0] * CONFIG["num_features"])
            cap.release()
            
            # Padding / Truncating
            seq_len = CONFIG["sequence_length"]
            if len(frame_features) > seq_len:
                padded_sequence = frame_features[:seq_len]
            else:
                padding = [[0.0] * CONFIG["num_features"]] * (seq_len - len(frame_features))
                padded_sequence = frame_features + padding
            
            label = get_label_from_filename(filename)
            if label != -1:
                all_features.append(padded_sequence)
                all_labels.append(label)
                logging.info(f"Successfully processed faces for {filename}")

        except Exception as e:
            logging.error(f"Could not process faces for {filename}. Error: {e}")
            continue

    logging.info(f"--- Saving Processed Facial Data ---")
    np.save(CONFIG["output_features"], np.array(all_features))
    np.save(CONFIG["output_labels"], np.array(all_labels))
    logging.info(f"Facial feature extraction complete. Data shape: {np.array(all_features).shape}")

# ==================================
# 5. SCRIPT EXECUTION
# ==================================
if __name__ == "__main__":
    process_facial_features()