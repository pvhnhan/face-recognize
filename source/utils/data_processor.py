"""
Module xử lý dữ liệu cho hệ thống nhận diện khuôn mặt
Chứa các hàm xử lý metadata, phân chia dữ liệu, và chuẩn bị dữ liệu huấn luyện YOLOv7
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.model_selection import train_test_split
import yaml

from config.config import DATA_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class DataProcessor:
    """Lớp xử lý dữ liệu cho hệ thống nhận diện khuôn mặt"""
    
    def __init__(self):
        """Khởi tạo DataProcessor với các đường dẫn từ config"""
        self.raw_images_dir = DATA_CONFIG['raw_images_dir']
        self.yolo_dataset_dir = DATA_CONFIG['yolo_dataset_dir']
        self.metadata_file = DATA_CONFIG['metadata_file']
        self.train_split = DATA_CONFIG['train_split']
        self.val_split = DATA_CONFIG['val_split']
        
        # Tạo thư mục nếu chưa tồn tại
        self._create_directories()
    
    def _create_directories(self):
        """Tạo các thư mục cần thiết cho dữ liệu YOLO"""
        directories = [
            self.raw_images_dir,
            self.yolo_dataset_dir,
            self.yolo_dataset_dir / 'images' / 'train',
            self.yolo_dataset_dir / 'images' / 'val',
            self.yolo_dataset_dir / 'images' / 'test',
            self.yolo_dataset_dir / 'labels' / 'train',
            self.yolo_dataset_dir / 'labels' / 'val',
            self.yolo_dataset_dir / 'labels' / 'test',
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Tạo thư mục: {directory}")
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Tải file metadata.csv chứa thông tin mapping ảnh với nhân viên
        
        Returns:
            pd.DataFrame: DataFrame chứa thông tin metadata
        """
        if not self.metadata_file.exists():
            logger.error(f"File metadata không tồn tại: {self.metadata_file}")
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        try:
            metadata = pd.read_csv(self.metadata_file)
            required_columns = ['filename', 'employee_id', 'full_name']
            
            # Kiểm tra các cột bắt buộc
            missing_columns = [col for col in required_columns if col not in metadata.columns]
            if missing_columns:
                raise ValueError(f"Thiếu các cột bắt buộc: {missing_columns}")
            
            logger.info(f"Đã tải metadata với {len(metadata)} records")
            return metadata
            
        except Exception as e:
            logger.error(f"Lỗi khi tải metadata: {e}")
            raise
    
    def validate_data_integrity(self) -> Dict[str, any]:
        """
        Kiểm tra tính toàn vẹn dữ liệu
        
        Returns:
            Dict: Thông tin về dữ liệu hợp lệ và không hợp lệ
        """
        metadata = self.load_metadata()
        valid_files = []
        invalid_files = []
        missing_images = []
        
        for _, row in metadata.iterrows():
            filename = row['filename']
            image_path = self.raw_images_dir / filename
            
            # Kiểm tra file ảnh
            if not image_path.exists():
                missing_images.append(filename)
                continue
            
            # Kiểm tra định dạng file
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                invalid_files.append(filename)
                continue
            
            valid_files.append(filename)
        
        result = {
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'missing_images': missing_images,
            'total_valid': len(valid_files),
            'total_invalid': len(invalid_files),
            'total_missing': len(missing_images),
            'total_files': len(metadata)
        }
        
        logger.info(f"Kết quả kiểm tra dữ liệu: {result}")
        return result
    
    def split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Phân chia dữ liệu thành tập huấn luyện và validation
        
        Args:
            test_size: Tỷ lệ dữ liệu validation
            
        Returns:
            Tuple: (train_metadata, val_metadata)
        """
        metadata = self.load_metadata()
        
        # Lọc chỉ những file hợp lệ
        valid_files = []
        for _, row in metadata.iterrows():
            filename = row['filename']
            image_path = self.raw_images_dir / filename
            
            if image_path.exists() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                valid_files.append(row)
        
        valid_metadata = pd.DataFrame(valid_files)
        
        if len(valid_metadata) == 0:
            raise ValueError("Không có dữ liệu hợp lệ để huấn luyện")
        
        # Kiểm tra số lượng samples per class
        employee_counts = valid_metadata['employee_id'].value_counts()
        total_samples = len(valid_metadata)
        num_classes = len(employee_counts)
        
        logger.info(f"Số lượng ảnh per employee: {employee_counts.to_dict()}")
        logger.info(f"Tổng số ảnh: {total_samples}, Số classes: {num_classes}")
        
        # Tính toán test_size thực tế
        actual_test_size = int(total_samples * test_size)
        logger.info(f"Test size yêu cầu: {actual_test_size} ảnh")
        
        # Kiểm tra điều kiện cho stratified split
        if actual_test_size < num_classes:
            logger.warning(f"Test size ({actual_test_size}) nhỏ hơn số classes ({num_classes})")
            logger.warning("Sẽ sử dụng random split thay vì stratified split")
            
            # Sử dụng random split
            train_metadata, val_metadata = train_test_split(
                valid_metadata, 
                test_size=test_size, 
                random_state=42
            )
        else:
            # Kiểm tra xem có employee nào có ít hơn 2 ảnh không
            insufficient_employees = employee_counts[employee_counts < 2]
            if len(insufficient_employees) > 0:
                logger.warning(f"Các employee có ít hơn 2 ảnh: {insufficient_employees.to_dict()}")
                logger.warning("Sẽ sử dụng random split thay vì stratified split")
                
                # Sử dụng random split nếu có employee có ít hơn 2 ảnh
                train_metadata, val_metadata = train_test_split(
                    valid_metadata, 
                    test_size=test_size, 
                    random_state=42
                )
            else:
                # Sử dụng stratified split nếu tất cả điều kiện đều thỏa mãn
                logger.info("Sử dụng stratified split")
                train_metadata, val_metadata = train_test_split(
                    valid_metadata, 
                    test_size=test_size, 
                    random_state=42,
                    stratify=valid_metadata['employee_id']
                )
        
        logger.info(f"Phân chia dữ liệu: {len(train_metadata)} train, {len(val_metadata)} validation")
        return train_metadata, val_metadata
    
    def prepare_yolo_dataset(self, output_dir: Path = None) -> Dict[str, str]:
        """
        Chuẩn bị dữ liệu theo format YOLO cho YOLOv7 chính thức
        
        Args:
            output_dir: Thư mục output cho dataset YOLO (mặc định là yolo_dataset_dir)
            
        Returns:
            Dict: Đường dẫn đến các file cấu hình YOLO
        """
        if output_dir is None:
            output_dir = self.yolo_dataset_dir
        
        train_metadata, val_metadata = self.split_data()
        
        # Tạo cấu trúc thư mục YOLO
        yolo_dirs = {
            'images': {
                'train': output_dir / 'images' / 'train',
                'val': output_dir / 'images' / 'val'
            },
            'labels': {
                'train': output_dir / 'labels' / 'train',
                'val': output_dir / 'labels' / 'val'
            }
        }
        
        # Tạo thư mục
        for split_dirs in yolo_dirs.values():
            for dir_path in split_dirs.values():
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Copy dữ liệu huấn luyện
        self._copy_split_data(train_metadata, yolo_dirs['images']['train'], yolo_dirs['labels']['train'])
        
        # Copy dữ liệu validation
        self._copy_split_data(val_metadata, yolo_dirs['images']['val'], yolo_dirs['labels']['val'])
        
        # Tạo file cấu hình YOLO
        config_path = self._create_yolo_config(output_dir, len(train_metadata), len(val_metadata))
        
        logger.info(f"Đã chuẩn bị dataset YOLO tại: {output_dir}")
        return {'config_path': str(config_path)}
    
    def _copy_split_data(self, metadata: pd.DataFrame, images_dir: Path, labels_dir: Path):
        """Copy dữ liệu cho một split cụ thể"""
        for _, row in metadata.iterrows():
            filename = row['filename']
            image_path = self.raw_images_dir / filename
            
            if image_path.exists():
                # Copy ảnh
                dest_image_path = images_dir / filename
                shutil.copy2(image_path, dest_image_path)
                
                # Tạo label YOLO (giả định tất cả ảnh đều có khuôn mặt)
                # Format YOLO: class_id center_x center_y width height
                # Với class_id = 0 cho face
                label_filename = f"{Path(filename).stem}.txt"
                label_path = labels_dir / label_filename
                
                # Tạo label mặc định (toàn bộ ảnh là khuôn mặt)
                # Đây là placeholder, trong thực tế cần annotation thực
                with open(label_path, 'w') as f:
                    f.write("0 0.5 0.5 0.8 0.8\n")  # Face ở giữa ảnh
    
    def _create_yolo_config(self, output_dir: Path, train_count: int, val_count: int) -> Path:
        """
        Tạo file cấu hình YOLO theo format chính thức
        
        Args:
            output_dir: Thư mục dataset
            train_count: Số lượng ảnh training
            val_count: Số lượng ảnh validation
            
        Returns:
            Path: Đường dẫn đến file config
        """
        config = {
            'nc': 1,  # Số classes (chỉ có face)
            'names': ['face'],  # Tên class
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test'
        }
        
        config_dir = Path('config/yolo')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = config_dir / 'face_detection.yaml'
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Đã tạo YOLO config: {config_path}")
        logger.info(f"Dataset info: {train_count} train, {val_count} val")
        
        return config_path
    
    def create_face_embeddings_dataset(self) -> Dict[str, str]:
        """
        Tạo dataset embeddings cho face recognition
        
        Returns:
            Dict: Thông tin về dataset embeddings
        """
        metadata = self.load_metadata()
        embeddings_info = {
            'total_images': len(metadata),
            'unique_employees': len(metadata['employee_id'].unique()),
            'embeddings_dir': str(MODEL_CONFIG['face_embeddings_dir'])
        }
        
        logger.info(f"Dataset embeddings: {embeddings_info}")
        return embeddings_info
    
    def get_employee_statistics(self) -> Dict:
        """
        Lấy thống kê về nhân viên trong dataset
        
        Returns:
            Dict: Thống kê nhân viên
        """
        metadata = self.load_metadata()
        
        stats = {
            'total_employees': len(metadata['employee_id'].unique()),
            'total_images': len(metadata),
            'images_per_employee': metadata.groupby('employee_id').size().to_dict(),
            'employee_list': metadata['employee_id'].unique().tolist()
        }
        
        logger.info(f"Employee statistics: {stats}")
        return stats 