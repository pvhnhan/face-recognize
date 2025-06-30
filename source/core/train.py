"""
Training script cho hệ thống nhận diện khuôn mặt
Sử dụng YOLOv7 cho face detection và DeepFace cho face recognition
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Thêm đường dẫn để import các module
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_processor import DataProcessor
from utils.face_utils import FaceProcessor
from utils.image_utils import ImageProcessor
from config.config import TRAINING_CONFIG, MODEL_CONFIG, DATA_CONFIG

# Thiết lập logging
logging.basicConfig(
    level=getattr(logging, TRAINING_CONFIG['logging']['level']),
    format=TRAINING_CONFIG['logging']['format'],
    handlers=[
        logging.FileHandler(TRAINING_CONFIG['logging']['file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FaceRecognitionTrainer:
    """
    Lớp huấn luyện hệ thống nhận diện khuôn mặt
    """
    
    def __init__(self):
        """Khởi tạo trainer"""
        self.data_processor = DataProcessor()
        self.face_processor = FaceProcessor()
        self.image_processor = ImageProcessor()
        
        # Đường dẫn
        self.data_dir = Path(DATA_CONFIG['data_dir'])
        self.models_dir = Path(MODEL_CONFIG['models_dir'])
        self.yolo_dataset_dir = self.data_dir / 'yolo_dataset'
        self.embeddings_dir = self.models_dir / 'face_embeddings'
        
        # Tạo thư mục nếu chưa có
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Khởi tạo Face Recognition Trainer")
    
    def prepare_dataset(self) -> bool:
        """
        Chuẩn bị dataset cho training
        
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info("Bắt đầu chuẩn bị dataset...")
            
            # Kiểm tra metadata
            metadata_path = self.data_dir / 'metadata.csv'
            if not metadata_path.exists():
                logger.error("Không tìm thấy metadata.csv")
                return False
            
            # Đọc metadata để kiểm tra
            metadata_df = pd.read_csv(metadata_path)
            logger.info(f"Đọc được {len(metadata_df)} records từ metadata")
            
            # Chuẩn bị YOLO dataset (không cần truyền metadata_df)
            success = self.data_processor.prepare_yolo_dataset(
                output_dir=self.yolo_dataset_dir
            )
            
            if success:
                logger.info("Chuẩn bị dataset thành công!")
                return True
            else:
                logger.error("Chuẩn bị dataset thất bại!")
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dataset: {e}")
            return False
    
    def create_embeddings(self) -> bool:
        """
        Tạo embeddings cho tất cả khuôn mặt trong dataset
        
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info("Bắt đầu tạo embeddings...")
            
            # Đọc metadata
            metadata_path = self.data_dir / 'metadata.csv'
            metadata_df = pd.read_csv(metadata_path)
            
            # Tạo embeddings mapping
            embeddings_mapping = {}
            total_faces = 0
            successful_faces = 0
            
            for idx, row in metadata_df.iterrows():
                try:
                    image_path = self.data_dir / 'raw_images' / row['filename']
                    if not image_path.exists():
                        logger.warning(f"Không tìm thấy ảnh: {image_path}")
                        continue
                    
                    # Đọc ảnh
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Không thể đọc ảnh: {image_path}")
                        continue
                    
                    # Phát hiện khuôn mặt
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) == 0:
                        logger.warning(f"Không tìm thấy khuôn mặt trong: {image_path}")
                        continue
                    
                    # Lấy khuôn mặt đầu tiên
                    x, y, w, h = faces[0]
                    face_region = image[y:y+h, x:x+w]
                    
                    # Trích xuất embedding
                    face_embedding = self.face_processor.extract_face_embedding(face_region)
                    
                    if face_embedding is not None:
                        # Lưu embedding
                        embedding_key = f"{row['employee_id']}_{row['filename']}"
                        embeddings_mapping[embedding_key] = {
                            'employee_id': row['employee_id'],
                            'full_name': row['full_name'],
                            'image_path': str(image_path),
                            'embedding': face_embedding.tolist(),
                            'bbox': [x, y, w, h],
                            'created_at': datetime.now().isoformat()
                        }
                        successful_faces += 1
                        logger.info(f"Tạo embedding thành công cho: {row['full_name']} ({row['employee_id']})")
                    
                    total_faces += 1
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý ảnh {row['filename']}: {e}")
                    continue
            
            # Lưu embeddings mapping
            embeddings_file = self.embeddings_dir / 'embeddings_mapping.json'
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings_mapping, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Hoàn thành tạo embeddings: {successful_faces}/{total_faces} khuôn mặt thành công")
            logger.info(f"Lưu embeddings tại: {embeddings_file}")
            
            return successful_faces > 0
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings: {e}")
            return False
    
    def train_yolov7(self) -> bool:
        """
        Huấn luyện YOLOv7 model cho face detection
        
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info("Bắt đầu huấn luyện YOLOv7...")
            
            # Kiểm tra YOLOv7 repository
            yolov7_path = Path(__file__).parent.parent / 'yolov7'
            if not yolov7_path.exists():
                logger.error("Không tìm thấy YOLOv7 repository")
                return False
            
            # Kiểm tra dataset
            if not self.yolo_dataset_dir.exists():
                logger.error("Không tìm thấy YOLO dataset")
                return False
            
            # Tạo file config cho YOLOv7
            data_yaml = self._create_yolo_data_config()
            if not data_yaml:
                return False
            
            # Chuẩn bị lệnh training
            train_cmd = self._prepare_training_command(data_yaml)
            
            # Chạy training
            logger.info(f"Chạy lệnh training: {train_cmd}")
            
            import subprocess
            result = subprocess.run(
                train_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=yolov7_path
            )
            
            if result.returncode == 0:
                logger.info("Huấn luyện YOLOv7 thành công!")
                return True
            else:
                logger.error("Lỗi trong quá trình training:")
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện YOLOv7: {e}")
            return False
    
    def _create_yolo_data_config(self) -> Optional[Path]:
        """
        Tạo file config cho YOLOv7 training
        
        Returns:
            Path: Đường dẫn đến file config
        """
        try:
            # Đếm số lượng ảnh trong từng split
            train_dir = self.yolo_dataset_dir / 'images' / 'train'
            val_dir = self.yolo_dataset_dir / 'images' / 'val'
            
            train_count = len(list(train_dir.glob('*.jpg'))) + len(list(train_dir.glob('*.png')))
            val_count = len(list(val_dir.glob('*.jpg'))) + len(list(val_dir.glob('*.png')))
            
            # Tạo nội dung config
            config_content = f"""
# YOLOv7 Face Detection Dataset Config
path: {self.yolo_dataset_dir.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['face']  # class names

# Dataset info
train_count: {train_count}
val_count: {val_count}
total_count: {train_count + val_count}
"""
            
            # Lưu config
            config_file = self.yolo_dataset_dir / 'dataset.yaml'
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            logger.info(f"Tạo YOLO config tại: {config_file}")
            return config_file
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo YOLO config: {e}")
            return None
    
    def _prepare_training_command(self, data_yaml: Path) -> str:
        """
        Chuẩn bị lệnh training YOLOv7
        
        Args:
            data_yaml: Đường dẫn đến file config
            
        Returns:
            str: Lệnh training
        """
        # Sử dụng YOLOv7 tiny để training nhanh hơn
        model_config = 'cfg/training/yolov7-tiny.yaml'
        
        cmd_parts = [
            "python train.py",
            f"--data {data_yaml}",
            f"--cfg {model_config}",
            f"--epochs {TRAINING_CONFIG['epochs']}",
            f"--batch-size {TRAINING_CONFIG['batch_size']}",
            f"--img-size {TRAINING_CONFIG['img_size']}",
            "--device 0" if TRAINING_CONFIG.get('use_gpu', False) else "--device cpu",
            "--project ../models/trained",
            "--name face_detection",
            "--exist-ok",
            "--save-period 10"
        ]
        
        return " ".join(cmd_parts)
    
    def save_training_log(self, training_info: Dict):
        """
        Lưu thông tin training
        
        Args:
            training_info: Thông tin training
        """
        try:
            log_file = Path(TRAINING_CONFIG['logging']['file']).parent / 'training_log.txt'
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=== FACE RECOGNITION TRAINING LOG ===\n")
                f.write(f"Training time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Dataset prepared: {training_info.get('dataset_prepared', False)}\n")
                f.write(f"Embeddings created: {training_info.get('embeddings_created', False)}\n")
                f.write(f"YOLOv7 trained: {training_info.get('yolov7_trained', False)}\n")
                f.write(f"Total faces processed: {training_info.get('total_faces', 0)}\n")
                f.write(f"Successful embeddings: {training_info.get('successful_embeddings', 0)}\n")
                f.write("=" * 40 + "\n")
            
            logger.info(f"Lưu training log tại: {log_file}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu training log: {e}")
    
    def run_training_pipeline(self) -> bool:
        """
        Chạy toàn bộ pipeline training
        
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info("Bắt đầu pipeline training...")
            start_time = time.time()
            
            training_info = {
                'dataset_prepared': False,
                'embeddings_created': False,
                'yolov7_trained': False,
                'total_faces': 0,
                'successful_embeddings': 0
            }
            
            # Bước 1: Chuẩn bị dataset
            logger.info("=== BƯỚC 1: Chuẩn bị dataset ===")
            if self.prepare_dataset():
                training_info['dataset_prepared'] = True
                logger.info("✅ Chuẩn bị dataset thành công")
            else:
                logger.error("❌ Chuẩn bị dataset thất bại")
                return False
            
            # Bước 2: Tạo embeddings
            logger.info("=== BƯỚC 2: Tạo embeddings ===")
            if self.create_embeddings():
                training_info['embeddings_created'] = True
                logger.info("✅ Tạo embeddings thành công")
            else:
                logger.error("❌ Tạo embeddings thất bại")
                return False
            
            # Bước 3: Huấn luyện YOLOv7
            logger.info("=== BƯỚC 3: Huấn luyện YOLOv7 ===")
            if self.train_yolov7():
                training_info['yolov7_trained'] = True
                logger.info("✅ Huấn luyện YOLOv7 thành công")
            else:
                logger.warning("⚠️ Huấn luyện YOLOv7 thất bại (có thể sử dụng pretrained)")
            
            # Lưu training log
            training_time = time.time() - start_time
            training_info['training_time'] = training_time
            self.save_training_log(training_info)
            
            logger.info(f"Pipeline training hoàn thành trong {training_time:.2f} giây")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi trong training pipeline: {e}")
            return False

def main():
    """Hàm main để chạy training"""
    trainer = FaceRecognitionTrainer()
    
    # Chạy pipeline training
    success = trainer.run_training_pipeline()
    
    if success:
        logger.info("🎉 Training hoàn thành thành công!")
        print("Training hoàn thành thành công!")
    else:
        logger.error("❌ Training thất bại!")
        print("Training thất bại!")

if __name__ == '__main__':
    main() 