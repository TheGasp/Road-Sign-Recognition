import os
import time
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import cv2
import sys

# Import custom modules
from detection import get_detector, reload_detector
from settings import read_config, update_config, initialize_config, reset_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Creates necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs('./static/images', exist_ok=True)

# Initializes configuration
initialize_config()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files():
    """Clean up old uploaded and result files"""
    try:
        current_time = time.time()
        # We clean files older than 1 hour
        for folder in [UPLOAD_FOLDER, RESULT_FOLDER]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    if current_time - os.path.getctime(file_path) > 3600:  # 1 hour
                        os.remove(file_path)
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/recognize')
def recognize():
    """Dedicated recognition page"""
    return render_template('recognize.html')

@app.route('/settings')
def settings():
    """Settings management page"""
    return render_template('settings.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle image detection requests"""
    start_time = time.time()
    
    # Clean up old files periodically
    cleanup_old_files()
    
    # Validate request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported formats: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP'}), 400
    
    # Check file size (max 10MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 400
    
    image_path = None
    try:
        # Save uploaded file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)
        
        logger.info(f"Processing uploaded file: {filename}")
        
        # Get detector and perform detection
        detector = get_detector()
        detection_results = detector.detect_signs(image_path)
        
        if detection_results['success']:
            result_image_url = None
            
            # Save annotated image if detections found and enabled
            config = read_config()
            if detection_results['total_detections'] > 0 and config.get('save_annotated_images', True):
                result_filename = f"result_{filename}"
                result_path = os.path.join(RESULT_FOLDER, result_filename)
                
                # Converts RGB to BGR for OpenCV
                annotated_bgr = cv2.cvtColor(detection_results['annotated_image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(result_path, annotated_bgr)
                result_image_url = f"/static/results/{result_filename}"
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response_data = {
                'success': True,
                'detections': detection_results['detections'],
                'total_detections': detection_results['total_detections'],
                'processing_time': round(processing_time, 3),
                'inference_time': detection_results.get('inference_time', 0),
                'image_info': {
                    'original_size': detection_results.get('image_size'),
                    'filename': file.filename
                },
                'model_info': detection_results.get('model_info', {}),
                'message': f"Detection completed: {detection_results['total_detections']} traffic sign(s) found"
            }
            
            if result_image_url:
                response_data['image_url'] = result_image_url
            
            # Return different response based on detections
            if detection_results['total_detections'] > 0:
                # Get the highest confidence detection for main result
                best_detection = max(detection_results['detections'], key=lambda x: x['confidence'])
                response_data['class'] = best_detection['class']
                response_data['confidence'] = best_detection['confidence']
            else:
                response_data['message'] = "No traffic signs detected in the image"
            
            return jsonify(response_data)
            
        else:
            return jsonify({
                'success': False,
                'error': detection_results['error'],
                'message': 'Detection failed'
            }), 500
            
    except Exception as e:
        logger.error(f"Detection processing error: {e}")
        return jsonify({
            'success': False,
            'error': f"Processing error: {str(e)}",
            'message': 'An unexpected error occurred during processing'
        }), 500
    finally:
        # Clean up uploaded file
        try:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup uploaded file: {e}")

# Configuration API routes
@app.route('/get-config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config = read_config()
        return jsonify(config)
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return jsonify({'error': 'Failed to read configuration'}), 500

@app.route('/update-config', methods=['POST'])
def update_config_route():
    """Update configuration parameter"""
    try:
        data = request.get_json()
        key = data.get('key')
        value = data.get('value')
        
        if not key or value is None:
            return jsonify({'error': 'Missing key or value'}), 400
        
        # Validates and converts value types
        if key in ['confidence_threshold', 'iou_threshold']:
            try:
                value = float(value)
                if key == 'confidence_threshold' and not (0.0 <= value <= 1.0):
                    return jsonify({'error': 'Confidence threshold must be between 0.0 and 1.0'}), 400
                if key == 'iou_threshold' and not (0.0 <= value <= 1.0):
                    return jsonify({'error': 'IoU threshold must be between 0.0 and 1.0'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid float value'}), 400
                
        elif key in ['image_size', 'max_detections']:
            try:
                value = int(value)
                if key == 'image_size' and value < 32:
                    return jsonify({'error': 'Image size must be at least 32 pixels'}), 400
                if key == 'max_detections' and value < 1:
                    return jsonify({'error': 'Max detections must be at least 1'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid integer value'}), 400
                
        elif key == 'save_annotated_images':
            value = bool(value)
        
        # Update configuration
        if update_config(key, value):
            # Reload detector configuration
            detector = get_detector()
            detector.reload_config()
            
            logger.info(f"Configuration updated: {key} = {value}")
            return jsonify({'success': True, 'updated': {key: value}})
        else:
            return jsonify({'error': f'Invalid configuration key: {key}'}), 400
            
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'error': 'Failed to update configuration'}), 500

@app.route('/reset-config', methods=['POST'])
def reset_config_route():
    """Reset configuration to defaults"""
    try:
        reset_config()
        # Force reload detector with new config
        reload_detector()
        logger.info("Configuration reset to defaults")
        return jsonify({'success': True, 'message': 'Configuration reset to defaults'})
    except Exception as e:
        logger.error(f"Error resetting config: {e}")
        return jsonify({'error': 'Failed to reset configuration'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        detector = get_detector()
        if detector.model:
            return jsonify({
                'success': True,
                'model_classes': len(detector.model_names),
                'class_names': list(detector.model_names.values()) if detector.model_names else [],
                'device': str(detector.device),
                'model_path': detector.config['model_path']
            })
        else:
            return jsonify({'success': False, 'error': 'Model not loaded'})
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Traffic Sign Detection App")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Results folder: {RESULT_FOLDER}")
    
    # Initialize detector on startup
    try:
        detector = get_detector()
        logger.info("Detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
    
    port = 5000
    if '--port=5001' in sys.argv:
        port = 5001
    app.run(debug=True, host='0.0.0.0', port=port)