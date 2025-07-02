"""
Script tạo dữ liệu mẫu cho training - Phiên bản đơn giản
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import logging
from PIL import Image, ImageEnhance, ImageFilter
import random
import os
import sys

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
        elif augmentation_type == 'brightness_dark':
            # Giảm độ sáng
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)
        elif augmentation_type == 'contrast':
            # Tăng độ tương phản
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
        elif augmentation_type == 'contrast_low':
            # Giảm độ tương phản
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(0.8)
        elif augmentation_type == 'blur':
            # Làm mờ nhẹ
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        elif augmentation_type == 'blur_strong':
            # Làm mờ mạnh hơn
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif augmentation_type == 'noise':
            # Thêm noise
            img_array = np.array(image)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
            image = Image.fromarray(img_array)
        elif augmentation_type == 'noise_strong':
            # Thêm noise mạnh hơn
            img_array = np.array(image)
            noise = np.random.normal(0, 20, img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
            image = Image.fromarray(img_array)
        elif augmentation_type == 'rotation':
            # Xoay nhẹ
            angle = random.uniform(-10, 10)
            image = image.rotate(angle, expand=True)
        elif augmentation_type == 'rotation_strong':
            # Xoay mạnh hơn
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, expand=True)
        elif augmentation_type == 'flip_horizontal':
            # Lật ngang
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif augmentation_type == 'color_jitter':
            # Thay đổi màu sắc
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
        elif augmentation_type == 'sharpness':
            # Tăng độ sắc nét
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
        elif augmentation_type == 'saturation':
            # Thay đổi độ bão hòa
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
        else:
            # Random augmentation
            augmentations = ['brightness', 'brightness_dark', 'contrast', 'contrast_low', 
                           'blur', 'blur_strong', 'noise', 'noise_strong', 'rotation', 
                           'rotation_strong', 'flip_horizontal', 'color_jitter', 
                           'sharpness', 'saturation']
            aug_type = random.choice(augmentations)
            return create_augmented_image(image_path, output_path, aug_type)
        
        # Lưu ảnh
        image.save(output_path)
        logger.info(f"Đã tạo ảnh augmented: {output_path}")
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo ảnh augmented: {e}")

def process_real_data():
    """
    Xử lý dữ liệu thực từ metadata.csv
    """
    metadata_file = DATA_CONFIG['metadata_file']
    raw_images_dir = DATA_CONFIG['raw_images_dir']
    
    # Đọc metadata hiện tại
    if not metadata_file.exists():
        logger.error(f"File metadata không tồn tại: {metadata_file}")
        return None
    
    metadata = pd.read_csv(metadata_file)
    logger.info(f"Đọc metadata với {len(metadata)} ảnh")
    
    # Kiểm tra và sửa dữ liệu
    valid_metadata = []
    missing_images = []
    
    for _, row in metadata.iterrows():
        filename = row['filename']
        employee_id = row['employee_id']
        full_name = row['full_name']
        
        # Kiểm tra file ảnh có tồn tại không
        image_path = raw_images_dir / filename
        
        if image_path.exists():
            # Thêm extension nếu cần
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # Tìm file với extension
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    test_path = raw_images_dir / f"{filename}{ext}"
                    if test_path.exists():
                        filename = f"{filename}{ext}"
                        image_path = test_path
                        break
            
            valid_metadata.append({
                'filename': filename,
                'employee_id': employee_id,
                'full_name': full_name
            })
            logger.info(f"✅ {filename} - {employee_id} ({full_name})")
        else:
            missing_images.append(filename)
            logger.warning(f"❌ File không tồn tại: {filename}")
    
    # Thống kê
    valid_df = pd.DataFrame(valid_metadata)
    employee_counts = valid_df['employee_id'].value_counts()
    
    logger.info(f"\n=== THỐNG KÊ DỮ LIỆU ===")
    logger.info(f"Tổng số ảnh trong metadata: {len(metadata)}")
    logger.info(f"Số ảnh hợp lệ: {len(valid_df)}")
    logger.info(f"Số ảnh thiếu: {len(missing_images)}")
    logger.info(f"Số employee: {len(employee_counts)}")
    logger.info(f"Số ảnh per employee: {employee_counts.to_dict()}")
    
    if missing_images:
        logger.warning(f"Files thiếu: {missing_images}")
    
    # Lưu metadata đã sửa
    valid_df.to_csv(metadata_file, index=False)
    logger.info(f"Đã lưu metadata đã sửa tại: {metadata_file}")
    
    return valid_df

def create_minimal_dataset():
    """
    Tạo dataset tối thiểu với 5 ảnh cho mỗi employee (cho test nhanh)
    """
    metadata_file = DATA_CONFIG['metadata_file']
    raw_images_dir = DATA_CONFIG['raw_images_dir']
    
    # Đọc metadata hiện tại
    if metadata_file.exists():
        metadata = pd.read_csv(metadata_file)
        logger.info(f"Đọc metadata hiện tại với {len(metadata)} ảnh")
    else:
        metadata = pd.DataFrame(columns=['filename', 'employee_id', 'full_name'])
        logger.info("Tạo metadata mới")
    
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
    
    # Mục tiêu: 5 ảnh cho mỗi employee
    target_per_employee = 5
    new_metadata = []
    
    for employee in employees:
        employee_id = employee['id']
        full_name = employee['name']
        
        logger.info(f"Xử lý employee {employee_id} ({full_name})")
        
        # Tạo ảnh gốc trước
        base_filename = f"{employee_id.lower()}.jpg"
        base_image_path = raw_images_dir / base_filename
        
        if not base_image_path.exists():
            # Tạo ảnh mẫu đơn giản
            img = np.ones((224, 224, 3), dtype=np.uint8) * 128
            cv2.putText(img, employee_id, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, full_name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(str(base_image_path), img)
            logger.info(f"Đã tạo ảnh gốc: {base_image_path}")
        
        # Thêm ảnh gốc vào metadata
        new_metadata.append({
            'filename': base_filename,
            'employee_id': employee_id,
            'full_name': full_name
        })
        
        # Tạo ảnh augmented để đạt target_per_employee
        for i in range(1, target_per_employee):
            aug_filename = f"{employee_id.lower()}_min_{i:02d}.jpg"
            aug_image_path = raw_images_dir / aug_filename
            
            # Chọn loại augmentation đơn giản
            augmentation_types = ['brightness', 'contrast', 'blur', 'noise', 'rotation']
            aug_type = augmentation_types[i % len(augmentation_types)]
            
            # Tạo ảnh augmented
            create_augmented_image(base_image_path, aug_image_path, aug_type)
            
            new_metadata.append({
                'filename': aug_filename,
                'employee_id': employee_id,
                'full_name': full_name
            })
    
    # Lưu metadata mới
    new_metadata_df = pd.DataFrame(new_metadata)
    new_metadata_df.to_csv(metadata_file, index=False)
    
    # Thống kê
    employee_counts = new_metadata_df['employee_id'].value_counts()
    logger.info(f"Đã tạo dataset tối thiểu với {len(new_metadata_df)} ảnh")
    logger.info(f"Số lượng ảnh per employee: {employee_counts.to_dict()}")
    logger.info(f"Metadata được lưu tại: {metadata_file}")
    
    return new_metadata_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'minimal':
        logger.info("Tạo dataset tối thiểu (5 ảnh/employee)...")
        create_minimal_dataset()
    else:
        logger.info("Xử lý dữ liệu thực từ metadata.csv...")
        process_real_data() 