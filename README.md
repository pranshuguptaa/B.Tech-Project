Overview:

This project detects human emotions and stress levels using both facial and audio inputs. It combines the outputs of two models through late fusion to produce a final prediction.

Models Used:

Facial Emotion Detection: YOLOv11n

Audio Emotion Detection: Custom CNN (saved in .h5 format)

Method:

Facial expressions are analyzed using the YOLO model.

Audio signals are processed using the CNN model.

Both outputs are combined through late fusion to determine the final emotion and stress level.
