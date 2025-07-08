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
from deepface import DeepFace

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
                    
                    # Phát hiện khuôn mặt sử dụng DeepFace (mạnh hơn OpenCV)
                    try:
                        # Chuyển BGR sang RGB cho DeepFace
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Sử dụng DeepFace để phát hiện khuôn mặt
                        face_objs = DeepFace.extract_faces(
                            img_path=image_rgb,
                            detector_backend='opencv',
                            enforce_detection=False
                        )
                        
                        if len(face_objs) == 0:
                            # Fallback về OpenCV nếu DeepFace không phát hiện được
                            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                            
                            if len(faces) == 0:
                                logger.warning(f"Không tìm thấy khuôn mặt trong: {image_path}")
                                continue
                            
                            # Lấy khuôn mặt đầu tiên
                            x, y, w, h = faces[0]
                            face_region = image[y:y+h, x:x+w]
                        else:
                            # Sử dụng kết quả từ DeepFace
                            face_obj = face_objs[0]
                            face_region = face_obj['face']
                            # Chuyển về BGR để tương thích với code cũ
                            face_region = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
                            # Lấy bbox từ DeepFace
                            facial_area = face_obj['facial_area']
                            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    except Exception as e:
                        logger.warning(f"Lỗi DeepFace detection, fallback về OpenCV: {e}")
                        # Fallback về OpenCV
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
                            'bbox': [int(x), int(y), int(w), int(h)],
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
            
            # Cài đặt YOLOv7 dependencies nếu cần
            if not self._install_yolov7_dependencies(yolov7_path):
                logger.warning("Không thể cài đặt YOLOv7 dependencies, thử training với dependencies hiện tại")
            
            # Tạo file config cho YOLOv7
            data_yaml = self._create_yolo_data_config()
            if not data_yaml:
                return False
            
            # Kiểm tra dataset trước khi training
            if not self._validate_dataset():
                logger.error("Dataset validation thất bại")                                
            
            # Debug: Kiểm tra đường dẫn dataset
            logger.info(f"Dataset path: {self.yolo_dataset_dir}")
            logger.info(f"Dataset exists: {self.yolo_dataset_dir.exists()}")
            logger.info(f"Train images dir: {self.yolo_dataset_dir / 'images' / 'train'}")
            logger.info(f"Train images exists: {(self.yolo_dataset_dir / 'images' / 'train').exists()}")
            logger.info(f"Val images dir: {self.yolo_dataset_dir / 'images' / 'val'}")
            logger.info(f"Val images exists: {(self.yolo_dataset_dir / 'images' / 'val').exists()}")
            
            # Debug thêm: Kiểm tra từ thư mục YOLOv7
            if yolov7_path.exists():
                logger.info(f"YOLOv7 path: {yolov7_path}")
                logger.info(f"YOLOv7 exists: {yolov7_path.exists()}")
                
                # Kiểm tra đường dẫn tương đối từ YOLOv7
                relative_dataset_path = yolov7_path.parent / "data" / "yolo_dataset"
                logger.info(f"Relative dataset path from YOLO: {relative_dataset_path}")
                logger.info(f"Relative dataset exists: {relative_dataset_path.exists()}")
                
                if relative_dataset_path.exists():
                    train_path_from_yolo = relative_dataset_path / "images" / "train"
                    val_path_from_yolo = relative_dataset_path / "images" / "val"
                    logger.info(f"Train path from YOLO: {train_path_from_yolo}")
                    logger.info(f"Train path from YOLO exists: {train_path_from_yolo.exists()}")
                    logger.info(f"Val path from YOLO: {val_path_from_yolo}")
                    logger.info(f"Val path from YOLO exists: {val_path_from_yolo.exists()}")
                    
                    if train_path_from_yolo.exists():
                        train_files = list(train_path_from_yolo.glob("*"))
                        logger.info(f"Train files count: {len(train_files)}")
                        if train_files:
                            logger.info(f"First few train files: {[f.name for f in train_files[:3]]}")
                    
                    if val_path_from_yolo.exists():
                        val_files = list(val_path_from_yolo.glob("*"))
                        logger.info(f"Val files count: {len(val_files)}")
                        if val_files:
                            logger.info(f"First few val files: {[f.name for f in val_files[:3]]}")
            
            # Kiểm tra nội dung thư mục
            if self.yolo_dataset_dir.exists():
                logger.info(f"Dataset contents: {list(self.yolo_dataset_dir.iterdir())}")
                if (self.yolo_dataset_dir / 'images').exists():
                    logger.info(f"Images contents: {list((self.yolo_dataset_dir / 'images').iterdir())}")
                    if (self.yolo_dataset_dir / 'images' / 'train').exists():
                        train_files = list((self.yolo_dataset_dir / 'images' / 'train').glob('*'))
                        logger.info(f"Train files: {len(train_files)} files")
                        if len(train_files) > 0:
                            logger.info(f"First few train files: {[f.name for f in train_files[:5]]}")
            
            # Kiểm tra thư mục hiện tại và tạo symlink nếu cần
            import os
            current_dir = os.getcwd()
            logger.info(f"Current working directory: {current_dir}")
            
            # Tạo symlink từ yolov7/data đến dataset thực tế
            yolov7_data_dir = yolov7_path / 'data'
            if not yolov7_data_dir.exists():
                yolov7_data_dir.mkdir(parents=True, exist_ok=True)
            

            
            # Tạo symlink cho dataset
            dataset_symlink = yolov7_data_dir / 'yolo_dataset'
            if dataset_symlink.exists():
                if dataset_symlink.is_symlink():
                    dataset_symlink.unlink()
                else:
                    import shutil
                    shutil.rmtree(dataset_symlink)
            
            try:
                import os
                os.symlink(str(self.yolo_dataset_dir), str(dataset_symlink))
                logger.info(f"Created symlink: {dataset_symlink} -> {self.yolo_dataset_dir}")
                
                # Không copy config vì symlink đã trỏ đến dataset gốc
                logger.info("Using symlinked dataset, no need to copy config")
                
            except Exception as e:
                logger.warning(f"Could not create symlink: {e}")
                # Fallback: copy dataset
                import shutil
                if dataset_symlink.exists():
                    shutil.rmtree(dataset_symlink)
                shutil.copytree(str(self.yolo_dataset_dir), str(dataset_symlink))
                logger.info(f"Copied dataset to: {dataset_symlink}")
                
                # Copy file config vào thư mục YOLOv7 chỉ khi copy dataset
                config_in_yolov7 = yolov7_data_dir / 'yolo_dataset' / 'dataset.yaml'
                if data_yaml.exists():
                    shutil.copy2(str(data_yaml), str(config_in_yolov7))
                    logger.info(f"Copied config to: {config_in_yolov7}")
            
            # Chuẩn bị lệnh training
            train_cmd = self._prepare_training_command(data_yaml)
            
            # Debug: Kiểm tra file config
            logger.info(f"Config file: {data_yaml}")
            logger.info(f"Config exists: {data_yaml.exists()}")
            if data_yaml.exists():
                with open(data_yaml, 'r') as f:
                    config_content = f.read()
                    logger.info(f"Config content:\n{config_content}")
                
                # Test đọc YAML -- Lỗi đọc file train và val
                import yaml
                try:
                    with open(data_yaml, 'r') as f:
                        test_data_dict = yaml.load(f, Loader=yaml.SafeLoader)
                    logger.info(f"YAML parsed successfully: {test_data_dict}")
                    
                    # Kiểm tra các đường dẫn trong data_dict
                    if 'train' in test_data_dict:
                        train_path = Path(test_data_dict['path']) / test_data_dict['train']
                        logger.info(f"Train path: {train_path}")
                        logger.info(f"Train path exists: {train_path.exists()}")
                    
                    if 'val' in test_data_dict:
                        val_path = Path(test_data_dict['path']) / test_data_dict['val']
                        logger.info(f"Val path: {val_path}")
                        logger.info(f"Val path exists: {val_path.exists()}")
                        
                except Exception as e:
                    logger.error(f"Error parsing YAML: {e}")
            
            # Debug: Kiểm tra đường dẫn dataset trước khi training
            logger.info("🔍 Debug: Kiểm tra đường dẫn dataset...")
            import yaml
            with open(data_yaml, 'r') as f:
                data_dict = yaml.load(f, Loader=yaml.SafeLoader)
            
            logger.info(f"Dataset config: {data_dict}")
            
            # Kiểm tra đường dẫn validation
            val_path = Path(data_dict['path']) / data_dict['val']
            logger.info(f"Validation path: {val_path}")
            logger.info(f"Validation path exists: {val_path.exists()}")
            logger.info(f"Validation path absolute: {val_path.resolve()}")
            
            # Kiểm tra từ thư mục YOLOv7
            yolov7_val_path = yolov7_path / data_dict['path'] / data_dict['val']
            logger.info(f"YOLOv7 validation path: {yolov7_val_path}")
            logger.info(f"YOLOv7 validation path exists: {yolov7_val_path.exists()}")
            logger.info(f"YOLOv7 validation path absolute: {yolov7_val_path.resolve()}")
            
            # Debug thêm: Kiểm tra chính xác như YOLOv7 check_dataset
            logger.info("🔍 Debug: Kiểm tra như YOLOv7 check_dataset...")
            val = data_dict.get('val')
            logger.info(f"Val from config: {val}")
            
            if val and len(val):
                val_paths = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]
                logger.info(f"Val paths after resolve: {val_paths}")
                
                for i, path in enumerate(val_paths):
                    logger.info(f"Val path {i}: {path}")
                    logger.info(f"Val path {i} exists: {path.exists()}")
                    logger.info(f"Val path {i} absolute: {path.resolve()}")
                
                missing_paths = [str(x) for x in val_paths if not x.exists()]
                logger.info(f"Missing paths: {missing_paths}")
                
                if missing_paths:
                    logger.error(f"❌ Dataset not found! Missing: {missing_paths}")
                else:
                    logger.info("✅ All validation paths exist!")
            
            # Chạy training
            logger.info(f"Chạy lệnh training: {train_cmd}")
            
            import subprocess
            import sys
            
            # Sử dụng Python interpreter hiện tại thay vì python trong thư mục YOLOv7
            python_cmd = sys.executable
            train_cmd = train_cmd.replace("python train.py", f"{python_cmd} train.py")
            
            # Thêm YOLOv7 path vào PYTHONPATH
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{yolov7_path}:{env.get('PYTHONPATH', '')}"
            
            result = subprocess.run(
                train_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=yolov7_path,
                env=env
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
    

    
    def _install_yolov7_dependencies(self, yolov7_path: Path) -> bool:
        """
        Cài đặt dependencies cho YOLOv7
        
        Args:
            yolov7_path: Đường dẫn đến thư mục YOLOv7
            
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info("Cài đặt YOLOv7 dependencies...")
            
            # Kiểm tra file requirements.txt
            requirements_file = yolov7_path / 'requirements.txt'
            if not requirements_file.exists():
                logger.warning("Không tìm thấy requirements.txt trong YOLOv7")
                return False
            
            # Cài đặt dependencies
            import subprocess
            result = subprocess.run(
                f"pip install -r {requirements_file}",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Cài đặt YOLOv7 dependencies thành công")
                return True
            else:
                logger.warning(f"⚠️ Lỗi cài đặt dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi cài đặt YOLOv7 dependencies: {e}")
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
            
            # Tạo nội dung config với đường dẫn tuyệt đối để tránh lỗi resolve
            # Trong Docker container, yolov7 nằm ở /app/yolov7 và dataset ở /app/data/yolo_dataset
            yolov7_path = Path(__file__).parent.parent / 'yolov7'
            
            # Kiểm tra xem có đang chạy trong Docker container không
            if Path("/app").exists():
                # Trong Docker container - sử dụng đường dẫn tuyệt đối
                absolute_path = "/app/data/yolo_dataset"
                logger.info(f"Running in Docker container, using absolute path: {absolute_path}")
            else:
                # Trong môi trường local - sử dụng đường dẫn tuyệt đối
                absolute_path = str(self.yolo_dataset_dir.absolute())
                logger.info(f"Running in local environment, using absolute path: {absolute_path}")
            
            # Debug: Kiểm tra đường dẫn cuối cùng
            final_path = Path(absolute_path)
            logger.info(f"Final dataset path: {final_path}")
            logger.info(f"Final dataset path exists: {final_path.exists()}")
            
            if final_path.exists():
                train_final = final_path / "images" / "train"
                val_final = final_path / "images" / "val"
                logger.info(f"Final train path: {train_final}")
                logger.info(f"Final train path exists: {train_final.exists()}")
                logger.info(f"Final val path: {val_final}")
                logger.info(f"Final val path exists: {val_final.exists()}")
            else:
                logger.error(f"Final dataset path does not exist: {final_path}")
            
            config_content = f"""
# YOLOv7 Face Detection Dataset Config
path: {absolute_path}  # dataset root dir (absolute path)
train: {absolute_path}/images/train  # train images (absolute path)
val: {absolute_path}/images/val  # val images (absolute path)

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
    
    def _validate_dataset(self) -> bool:
        """
        Kiểm tra tính hợp lệ của dataset
        
        Returns:
            bool: True nếu dataset hợp lệ
        """
        try:
            logger.info("Kiểm tra dataset...")
            
            # Kiểm tra thư mục images
            train_images_dir = self.yolo_dataset_dir / 'images' / 'train'
            val_images_dir = self.yolo_dataset_dir / 'images' / 'val'
            
            if not train_images_dir.exists():
                logger.error(f"Không tìm thấy thư mục train images: {train_images_dir}")
                return False
            
            if not val_images_dir.exists():
                logger.error(f"Không tìm thấy thư mục val images: {val_images_dir}")
                return False
            
            # Kiểm tra thư mục labels
            train_labels_dir = self.yolo_dataset_dir / 'labels' / 'train'
            val_labels_dir = self.yolo_dataset_dir / 'labels' / 'val'
            
            if not train_labels_dir.exists():
                logger.error(f"Không tìm thấy thư mục train labels: {train_labels_dir}")
                return False
            
            if not val_labels_dir.exists():
                logger.error(f"Không tìm thấy thư mục val labels: {val_labels_dir}")
                return False
            
            # Đếm số lượng ảnh và labels
            train_images = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png'))
            val_images = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))
            
            train_labels = list(train_labels_dir.glob('*.txt'))
            val_labels = list(val_labels_dir.glob('*.txt'))
            
            logger.info(f"Train images: {len(train_images)}, Train labels: {len(train_labels)}")
            logger.info(f"Val images: {len(val_images)}, Val labels: {len(val_labels)}")
            
            # Kiểm tra xem có ít nhất 1 ảnh trong mỗi split không
            if len(train_images) == 0:
                logger.error("Không có ảnh training")
                return False
            
            if len(val_images) == 0:
                logger.error("Không có ảnh validation")
                return False
            
            # Kiểm tra xem số lượng ảnh và labels có khớp không
            if len(train_images) != len(train_labels):
                logger.warning(f"Số lượng train images ({len(train_images)}) không khớp với train labels ({len(train_labels)})")
            
            if len(val_images) != len(val_labels):
                logger.warning(f"Số lượng val images ({len(val_images)}) không khớp với val labels ({len(val_labels)})")
            
            logger.info("✅ Dataset validation thành công")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi validate dataset: {e}")
            return False
    
    def _prepare_training_command(self, data_yaml: Path) -> str:
        """
        Chuẩn bị lệnh training YOLOv7
        
        Args:
            data_yaml: Đường dẫn đến file config
            
        Returns:
            str: Lệnh training
        """
        # Sử dụng YOLOv7 tiny để training nhanh hơn
        # Sử dụng đường dẫn tuyệt đối để tránh lỗi resolve
        if Path("/app").exists():
            # Trong Docker container - sử dụng đường dẫn tuyệt đối
            absolute_data_yaml = "/app/data/yolo_dataset/dataset.yaml"
            model_config = "cfg/training/yolov7-tiny.yaml"
            project_path = "/app/models/trained"
        else:
            # Trong môi trường local - sử dụng đường dẫn tuyệt đối
            absolute_data_yaml = str(data_yaml.absolute())
            model_config = "cfg/training/yolov7-tiny.yaml"
            project_path = str(Path(__file__).parent.parent / "models/trained")
        
        cmd_parts = [
            "python train.py",
            f"--data {absolute_data_yaml}",
            f"--cfg {model_config}",
            "--weights ''",  # Không sử dụng pretrained weights để tránh lỗi download
            f"--epochs {TRAINING_CONFIG['epochs']}",
            f"--batch-size {TRAINING_CONFIG['batch_size']}",
            f"--img-size {TRAINING_CONFIG['img_size']}",
            "--device 0" if TRAINING_CONFIG.get('use_gpu', False) else "--device cpu",
            f"--project {project_path}",
            "--name face_detection",
            "--exist-ok",
            "--workers 0"  # Giảm workers để tránh lỗi shared memory
        ]
        
        final_cmd = " ".join(cmd_parts)
        logger.info(f"🔧 Generated training command: {final_cmd}")
        return final_cmd
    
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