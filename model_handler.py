import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
import cv2

class ImageClassifier:
    def __init__(self):
        # Load pre-trained MobileNetV2 model (lighter than VGG16)
        print("Loading classification model...")
        self.model = MobileNetV2(weights='imagenet')
        print("Model loaded successfully!")
        
    def classify_image(self, img):
        """Classify image and return top 5 predictions"""
        # Resize to model input size
        img_resized = cv2.resize(img, (224, 224))
        
        # Expand dimensions
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Preprocess
        img_preprocessed = preprocess_input(img_array)
        
        # Predict
        predictions = self.model.predict(img_preprocessed, verbose=0)
        
        # Decode predictions
        decoded = decode_predictions(predictions, top=5)[0]
        
        # Format results
        results = []
        for pred in decoded:
            results.append({
                'class': pred[1].replace('_', ' ').title(),
                'confidence': float(pred[2]) * 100
            })
        
        return results