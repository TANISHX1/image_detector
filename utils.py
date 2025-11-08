import requests
from PIL import Image
from io import BytesIO
import numpy as np
import cv2

def load_image_from_url(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        raise Exception(f"Error loading image from URL: {str(e)}")

def load_image_from_file(filepath):
    """Load image from local file"""
    try:
        img = Image.open(filepath)
        img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        raise Exception(f"Error loading image file: {str(e)}")

def resize_for_display(img, max_size=400):
    """Resize image for GUI display"""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
    return img