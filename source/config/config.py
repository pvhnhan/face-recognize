"""
Cấu hình hệ thống nhận diện khuôn mặt với YOLOv7 chính thức
Chứa các tham số quan trọng cho việc huấn luyện và suy luận
"""

import os
from pathlib import Path

# Đường dẫn gốc của dự án
BASE_DIR = Path(__file__).parent.parent

# Cấu hình dữ liệu
DATA_CONFIG = {
    'data_dir': BASE_DIR / 'data',
    'raw_images_dir': BASE_DIR / 'data' / 'raw_images',
    'yolo_dataset_dir': BASE_DIR / 'data' / 'yolo_dataset',
    'metadata_file': BASE_DIR / 'data' / 'metadata.csv',
    'train_split': 0.8,  # Tỷ lệ dữ liệu huấn luyện
    'val_split': 0.2,    # Tỷ lệ dữ liệu validation
}

# Cấu hình mô hình YOLOv7 chính thức
MODEL_CONFIG = {
    'models_dir': BASE_DIR / 'models',
    # YOLOv7 pretrained weights (từ repository chính thức)
    'yolov7_weights': BASE_DIR / 'models' / 'pretrained' / 'yolov7.pt',
    # Custom trained weights (sau khi training)
    'trained_weights': BASE_DIR / 'models' / 'trained' / 'face_detection' / 'weights' / 'best.pt',
    # Face embeddings cho recognition
    'face_embeddings_dir': BASE_DIR / 'models' / 'face_embeddings',
    # YOLOv7 detection parameters
    'confidence_threshold': 0.5,  # Ngưỡng tin cậy cho YOLO
    'nms_threshold': 0.4,         # Ngưỡng NMS
    'img_size': 640,              # Kích thước ảnh input cho YOLOv7
}

# Cấu hình nhận diện khuôn mặt
FACE_RECOGNITION_CONFIG = {
    'similarity_threshold': 0.6,  # Ngưỡng cosine similarity
    'embedding_model': 'Facenet512',  # Mô hình embedding (VGG-Face, Facenet, Facenet512, OpenFace, DeepID, ArcFace, SFace) - Facenet512 tốt hơn
    'detector_backend': 'opencv',   # Backend cho face detection (fallback)
    'face_size': (160, 160),     # Kích thước chuẩn hóa khuôn mặt
    'max_faces': 10,             # Số lượng khuôn mặt tối đa trong ảnh
}

# Cấu hình huấn luyện YOLOv7
TRAINING_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'file': '/app/logs/training.log'
    },
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.01,
    'img_size': 640,
    'device': 'auto',  # 'cpu', 'gpu', hoặc 'auto'
    'workers': 4,
    'patience': 20,    # Early stopping patience
    # YOLOv7 specific parameters
    'yolo_cfg': 'cfg/training/yolov7.yaml',
    'yolo_hyp': 'data/hyp.scratch.p5.yaml',
    'project': 'models/trained',
    'name': 'face_detection',
}

INFERENCE_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'file': 'logs/training.log'
    },
    'batch_size': 8,
    'device': 'auto',  # 'cpu', 'gpu', hoặc 'auto'
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    # Thêm các tham số khác nếu cần cho inference
}

# Cấu hình Flask API
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp'],
    'temp_dir': BASE_DIR / 'temp',
    'output_format': 'base64',  # 'base64' hoặc 'url'
}

# Cấu hình Docker
DOCKER_CONFIG = {
    'cpu_image': 'face-recognition-cpu:latest',
    'gpu_image': 'face-recognition-gpu:latest',
    'container_name': 'face-recognition-api',
    'gpu_base_image': 'nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04',
}

# Cấu hình logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': BASE_DIR / 'logs' / 'app.log',
}

# Cấu hình YOLOv7 repository
YOLOV7_CONFIG = {
    'repo_path': BASE_DIR / 'yolov7',
    'repo_url': 'https://github.com/WongKinYiu/yolov7.git',
    'required_files': [
        'train.py',
        'detect.py',
        'models/experimental.py',
        'utils/general.py',
        'utils/augmentations.py',
        'cfg/training/yolov7.yaml',
        'data/hyp.scratch.p5.yaml'
    ]
}

# Cấu hình môi trường ảo
VENV_CONFIG = {
    'venv_dir': BASE_DIR / 'venv',
    'python_path': BASE_DIR / 'venv' / 'bin' / 'python',
    'pip_path': BASE_DIR / 'venv' / 'bin' / 'pip',
}

# Tạo các thư mục cần thiết
def create_directories():
    """Tạo các thư mục cần thiết cho hệ thống"""
    directories = [
        # Data directories
        DATA_CONFIG['raw_images_dir'],
        DATA_CONFIG['yolo_dataset_dir'],
        DATA_CONFIG['yolo_dataset_dir'] / 'images' / 'train',
        DATA_CONFIG['yolo_dataset_dir'] / 'images' / 'val',
        DATA_CONFIG['yolo_dataset_dir'] / 'images' / 'test',
        DATA_CONFIG['yolo_dataset_dir'] / 'labels' / 'train',
        DATA_CONFIG['yolo_dataset_dir'] / 'labels' / 'val',
        DATA_CONFIG['yolo_dataset_dir'] / 'labels' / 'test',
        
        # Model directories
        BASE_DIR / 'models' / 'pretrained',
        BASE_DIR / 'models' / 'trained',
        MODEL_CONFIG['face_embeddings_dir'],
        
        # Output directories
        BASE_DIR / 'outputs',
        FLASK_CONFIG['temp_dir'],
        BASE_DIR / 'logs',
        
        # Config directories
        BASE_DIR / 'config' / 'yolo',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Khởi tạo thư mục khi import module
create_directories() 