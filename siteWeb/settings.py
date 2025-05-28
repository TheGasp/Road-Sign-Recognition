import os
import json

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "image_size": 640,
    "model_path": "../best.pt",
    "max_detections": 10,
    "save_annotated_images": True
}

def initialize_config():
    """Initialize configuration file with default values if it doesn't exist"""
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
            print("Configuration file created with default values.")
    else:
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            
            # Check for missing keys and add them
            missing_keys = [key for key in DEFAULT_CONFIG if key not in config]
            if missing_keys:
                print(f"Adding missing parameters: {missing_keys}")
                for key in missing_keys:
                    config[key] = DEFAULT_CONFIG[key]
                
                with open(CONFIG_FILE, "w") as f:
                    json.dump(config, f, indent=4)
                    
        except (json.JSONDecodeError, ValueError):
            # Recreate file if corrupted
            with open(CONFIG_FILE, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
                print("Configuration file recreated with default values.")

def read_config():
    """Read current configuration"""
    initialize_config()
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def update_config(key, value):
    """Update a specific configuration parameter"""
    config = read_config()
    if key in config:
        config[key] = value
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        return True
    return False

def reset_config():
    """Reset configuration to default values"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    return True