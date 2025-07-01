"""
Script tạo dữ liệu mẫu cho training
Tạo thêm ảnh cho các employee có ít dữ liệu
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from PIL import Image, ImageEnhance, ImageFilter
import random

import sys
from pathlib import Path

# Thêm thư mục gốc vào Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import DATA_CONFIG

logger = logging.getLogger(__name__)

def create_augmented_image(image_path: Path, output_path: Path, augmentation_type: str = 'random'):
    """
    Tạo ảnh augmented từ ảnh gốc
    
    Args:
        image_path: Đường dẫn ảnh gốc
        output_path: Đường dẫn ảnh output
        augmentation_type: Loại augmentation
    """
    try:
        # Đọc ảnh
        image = Image.open(image_path)
        
        if augmentation_type == 'brightness':
            # Tăng độ sáng
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.3)
        elif augmentation_type == 'contrast':
            # Tăng độ tương phản
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
        elif augmentation_type == 'blur':
            # Làm mờ nhẹ
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        elif augmentation_type == 'noise':
            # Thêm noise
            img_array = np.array(image)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
            image = Image.fromarray(img_array)
        elif augmentation_type == 'rotation':
            # Xoay nhẹ
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, expand=True)
        else:
            # Random augmentation
            augmentations = ['brightness', 'contrast', 'blur', 'noise', 'rotation']
            aug_type = random.choice(augmentations)
            return create_augmented_image(image_path, output_path, aug_type)
        
        # Lưu ảnh
        image.save(output_path)
        logger.info(f"Đã tạo ảnh augmented: {output_path}")
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo ảnh augmented: {e}")

def create_sample_dataset():
    """
    Tạo dataset mẫu với đủ dữ liệu cho training
    """
    metadata_file = DATA_CONFIG['metadata_file']
    raw_images_dir = DATA_CONFIG['raw_images_dir']
    
    # Đọc metadata hiện tại
    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file)
    else:
        # Tạo metadata mới nếu chưa có
        metadata = pd.DataFrame(columns=['filename', 'employee_id', 'full_name'])
    
    # Tạo thư mục raw_images nếu chưa có
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo ảnh mẫu cho mỗi employee
    employees = [
        {'id': 'E001', 'name': 'Dĩnh Anh'},
        {'id': 'E002', 'name': 'Doãn Minh'},
        {'id': 'E003', 'name': 'Trần Đức Nhân'},
        {'id': 'E004', 'name': 'Tài'},
        {'id': 'E005', 'name': 'Quang Quân'},
        {'id': 'E006', 'name': 'Minh Hà'}
    ]
    
    new_metadata = []
    
    for employee in employees:
        employee_id = employee['id']
        full_name = employee['name']
        
        # Tạo ít nhất 3 ảnh cho mỗi employee
        for i in range(3):
            if i == 0:
                # Ảnh gốc (nếu có)
                filename = f"{employee_id.lower()}.jpg"
                image_path = raw_images_dir / filename
                
                if not image_path.exists():
                    # Tạo ảnh mẫu đơn giản
                    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
                    cv2.putText(img, employee_id, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, full_name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.imwrite(str(image_path), img)
                
                new_metadata.append({
                    'filename': filename,
                    'employee_id': employee_id,
                    'full_name': full_name
                })
            else:
                # Ảnh augmented
                base_filename = f"{employee_id.lower()}.jpg"
                base_image_path = raw_images_dir / base_filename
                
                if base_image_path.exists():
                    # Tạo ảnh augmented
                    aug_filename = f"{employee_id.lower()}_aug_{i}.jpg"
                    aug_image_path = raw_images_dir / aug_filename
                    
                    create_augmented_image(base_image_path, aug_image_path, f'aug_{i}')
                    
                    new_metadata.append({
                        'filename': aug_filename,
                        'employee_id': employee_id,
                        'full_name': full_name
                    })
    
    # Lưu metadata mới
    new_metadata_df = pd.DataFrame(new_metadata)
    new_metadata_df.to_csv(metadata_file, index=False)
    
    logger.info(f"Đã tạo dataset mẫu với {len(new_metadata_df)} ảnh")
    logger.info(f"Metadata được lưu tại: {metadata_file}")
    
    return new_metadata_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_sample_dataset() 