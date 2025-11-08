import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

class FeatureExtractor:
    def __init__(self):
        # Load pre-trained VGG16 model for deep features
        self.deep_model = VGG16(weights='imagenet', include_top=False, 
                                pooling='avg')
    
    def extract_color_histogram(self, img):
        """Extract color histogram features"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate all histograms
        color_features = np.concatenate([hist_h, hist_s, hist_v])
        
        return color_features
    
    def extract_edge_features(self, img):
        """Extract edge-based features using Canny"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density in grid cells
        h, w = edges.shape
        cell_h, cell_w = h // 4, w // 4
        
        edge_features = []
        for i in range(4):
            for j in range(4):
                cell = edges[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                edge_density = np.sum(cell > 0) / (cell_h * cell_w)
                edge_features.append(edge_density)
        
        return np.array(edge_features)
    
    def extract_deep_features(self, img):
        """Extract deep learning features using VGG16"""
        # Resize image to 224x224 (VGG16 input size)
        img_resized = cv2.resize(img, (224, 224))
        
        # Expand dimensions for batch
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Preprocess for VGG16
        img_preprocessed = preprocess_input(img_array)
        
        # Extract features
        features = self.deep_model.predict(img_preprocessed, verbose=0)
        
        return features.flatten()
    
    def extract_all_features(self, img):
        """Extract all features and return as dictionary"""
        features = {
            'color_histogram': self.extract_color_histogram(img),
            'edge_features': self.extract_edge_features(img),
            'deep_features': self.extract_deep_features(img)
        }
        return features