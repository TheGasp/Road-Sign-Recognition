import os
import sys
import torch
from PIL import Image
import cv2
import numpy as np
import platform
import logging
from datetime import datetime

# Add YOLOv5 to path
sys.path.append("yolov5")

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device
from settings import read_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficSignDetector:
    def __init__(self):
        """Initialize the detector with current configuration"""
        self.config = read_config()
        self.device = select_device('')
        self.model = None
        self.model_names = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv5 model with Windows compatibility fix"""
        import pathlib
        temp = pathlib.PosixPath
        
        # Windows compatibility fix
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
        
        try:
            model_path = self.config['model_path']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = DetectMultiBackend(model_path, device=self.device)
            self.model.eval()
            self.model_names = self.model.names
            logger.info(f"Model loaded successfully from {model_path}")
            logger.info(f"Model classes: {len(self.model_names)} classes loaded")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        finally:
            pathlib.PosixPath = temp
    
    def reload_config(self):
        """Reload configuration and reinitialize model if needed"""
        old_model_path = self.config.get('model_path')
        self.config = read_config()
        
        # Reload model if path changed
        if old_model_path != self.config.get('model_path'):
            logger.info("Model path changed, reloading model...")
            self._load_model()
    
    def detect_signs(self, image_path):
        """
        Perform traffic sign detection on an image
        Returns: dict with detection results
        """
        if not self.model:
            return {
                'success': False,
                'error': 'Model not loaded',
                'detections': [],
                'total_detections': 0
            }
            
        try:
            # Reload config for latest settings
            self.reload_config()
            
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            
            # Preprocess image
            img_size = self.config['image_size']
            img_resized = img.resize((img_size, img_size))
            img_array = np.array(img_resized)
            img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).div(255.0).unsqueeze(0).to(self.device)
            
            # Perform inference
            logger.info(f"Running inference on image: {os.path.basename(image_path)}")
            start_time = datetime.now()
            
            with torch.no_grad():
                pred = self.model(img_tensor, augment=False)
                pred = non_max_suppression(
                    pred, 
                    conf_thres=self.config['confidence_threshold'], 
                    iou_thres=self.config['iou_threshold'],
                    max_det=self.config['max_detections']
                )
            
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Process results
            results = []
            img_annotated = np.array(img_resized).copy()
            
            if pred[0] is not None and len(pred[0]) > 0:
                for detection in pred[0]:
                    *xyxy, conf, cls = detection
                    
                    # Extract detection information
                    class_id = int(cls)
                    class_name = self.model_names[class_id] if class_id < len(self.model_names) else f"Class_{class_id}"
                    confidence = float(conf)
                    bbox = [int(x.item()) for x in xyxy]
                    
                    # Scale bbox back to original image size
                    scale_x = original_size[0] / img_size
                    scale_y = original_size[1] / img_size
                    original_bbox = [
                        int(bbox[0] * scale_x),
                        int(bbox[1] * scale_y),
                        int(bbox[2] * scale_x),
                        int(bbox[3] * scale_y)
                    ]
                    
                    results.append({
                        'class_id': class_id,
                        'class': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': bbox,  # For display image
                        'original_bbox': original_bbox  # For original image
                    })
                    
                    # Draw bounding box on resized image for display
                    if self.config['save_annotated_images']:
                        label = f"{class_name} {confidence:.2f}"
                        cv2.rectangle(img_annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        
                        # Calculate text size and background
                        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(img_annotated, 
                                    (bbox[0], bbox[1] - text_height - 10), 
                                    (bbox[0] + text_width, bbox[1]), 
                                    (0, 255, 0), -1)
                        cv2.putText(img_annotated, label, (bbox[0], bbox[1] - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            logger.info(f"Detection completed: {len(results)} signs found in {inference_time:.3f}s")
            
            return {
                'success': True,
                'detections': results,
                'annotated_image': img_annotated,
                'total_detections': len(results),
                'inference_time': round(inference_time, 3),
                'image_size': original_size,
                'model_info': {
                    'classes': len(self.model_names),
                    'device': str(self.device)
                }
            }
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'total_detections': 0
            }

# Global detector instance
_detector = None

def get_detector():
    """Get global detector instance (singleton pattern)"""
    global _detector
    if _detector is None:
        _detector = TrafficSignDetector()
    return _detector

def reload_detector():
    """Force reload of detector (useful when config changes)"""
    global _detector
    _detector = TrafficSignDetector()
    return _detector