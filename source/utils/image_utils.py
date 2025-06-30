"""
Module xử lý ảnh cho hệ thống nhận diện khuôn mặt
Chứa các hàm tiện ích để xử lý ảnh, chuyển đổi format, và visualization
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Lớp xử lý ảnh cho hệ thống nhận diện khuôn mặt"""
    
    def __init__(self):
        """Khởi tạo ImageProcessor"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Tải ảnh từ đường dẫn
        
        Args:
            image_path: Đường dẫn đến file ảnh
            
        Returns:
            np.ndarray: Ảnh dạng numpy array
        """
        try:
            image_path = Path(image_path)
            
            if not image_path.exists():
                raise FileNotFoundError(f"File ảnh không tồn tại: {image_path}")
            
            # Kiểm tra định dạng file
            if image_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Định dạng file không được hỗ trợ: {image_path.suffix}")
            
            # Đọc ảnh bằng OpenCV
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")
            
            logger.info(f"Đã tải ảnh thành công: {image_path}")
            return image
            
        except Exception as e:
            logger.error(f"Lỗi khi tải ảnh {image_path}: {e}")
            raise
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path], quality: int = 95):
        """
        Lưu ảnh ra file
        
        Args:
            image: Ảnh dạng numpy array
            output_path: Đường dẫn output
            quality: Chất lượng ảnh (1-100)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu ảnh bằng OpenCV
            success = cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if not success:
                raise ValueError(f"Không thể lưu ảnh: {output_path}")
            
            logger.info(f"Đã lưu ảnh thành công: {output_path}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu ảnh {output_path}: {e}")
            raise
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    keep_aspect_ratio: bool = True) -> np.ndarray:
        """
        Thay đổi kích thước ảnh
        
        Args:
            image: Ảnh input
            target_size: Kích thước mục tiêu (width, height)
            keep_aspect_ratio: Có giữ tỷ lệ khung hình không
            
        Returns:
            np.ndarray: Ảnh đã resize
        """
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            if keep_aspect_ratio:
                # Tính tỷ lệ scale
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # Resize ảnh
                resized = cv2.resize(image, (new_w, new_h))
                
                # Tạo ảnh mới với kích thước target và padding
                result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                
                # Tính vị trí để center ảnh
                y_offset = (target_h - new_h) // 2
                x_offset = (target_w - new_w) // 2
                
                result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                
            else:
                # Resize trực tiếp về kích thước target
                result = cv2.resize(image, target_size)
            
            logger.debug(f"Resize ảnh từ {w}x{h} về {target_w}x{target_h}")
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi resize ảnh: {e}")
            raise
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  padding: float = 0.1) -> np.ndarray:
        """
        Crop khuôn mặt từ ảnh
        
        Args:
            image: Ảnh gốc
            bbox: Bounding box (x, y, width, height)
            padding: Padding thêm xung quanh khuôn mặt (0.1 = 10%)
            
        Returns:
            np.ndarray: Ảnh khuôn mặt đã crop
        """
        try:
            h, w = image.shape[:2]
            x, y, face_w, face_h = bbox
            
            # Tính padding
            pad_x = int(face_w * padding)
            pad_y = int(face_h * padding)
            
            # Tính tọa độ crop với padding
            crop_x1 = max(0, x - pad_x)
            crop_y1 = max(0, y - pad_y)
            crop_x2 = min(w, x + face_w + pad_x)
            crop_y2 = min(h, y + face_h + pad_y)
            
            # Crop ảnh
            cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            logger.debug(f"Crop khuôn mặt: {bbox} -> {cropped.shape}")
            return cropped
            
        except Exception as e:
            logger.error(f"Lỗi khi crop khuôn mặt: {e}")
            raise
    
    def draw_face_boxes(self, image: np.ndarray, face_results: List[Dict], 
                       show_names: bool = True) -> np.ndarray:
        """
        Vẽ bounding box và thông tin lên ảnh
        
        Args:
            image: Ảnh gốc
            face_results: Kết quả nhận diện khuôn mặt
            show_names: Có hiển thị tên không
            
        Returns:
            np.ndarray: Ảnh đã vẽ
        """
        try:
            # Copy ảnh để không thay đổi ảnh gốc
            result_image = image.copy()
            
            for face_result in face_results:
                bbox = face_result['bbox']
                x, y, w, h = bbox
                
                # Xác định màu sắc dựa trên kết quả nhận diện
                if face_result.get('is_recognized', False):
                    color = (0, 255, 0)  # Xanh lá - nhận diện thành công
                    label = f"{face_result.get('full_name', 'Unknown')} ({face_result.get('similarity_score', 0):.2f})"
                else:
                    color = (0, 0, 255)  # Đỏ - không nhận diện được
                    label = "Unknown"
                
                # Vẽ bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                
                # Vẽ label nếu cần
                if show_names:
                    # Tính kích thước text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # Lấy kích thước text
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Vẽ background cho text
                    cv2.rectangle(result_image, 
                                (x, y - text_height - 10), 
                                (x + text_width, y), 
                                color, -1)
                    
                    # Vẽ text
                    cv2.putText(result_image, label, (x, y - 5), 
                              font, font_scale, (255, 255, 255), thickness)
            
            logger.debug(f"Đã vẽ {len(face_results)} bounding boxes")
            return result_image
            
        except Exception as e:
            logger.error(f"Lỗi khi vẽ bounding boxes: {e}")
            raise
    
    def image_to_base64(self, image: np.ndarray, format: str = 'JPEG', quality: int = 95) -> str:
        """
        Chuyển ảnh thành base64 string
        
        Args:
            image: Ảnh dạng numpy array
            format: Định dạng ảnh ('JPEG', 'PNG')
            quality: Chất lượng ảnh (1-100)
            
        Returns:
            str: Base64 string
        """
        try:
            # Chuyển BGR sang RGB
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Chuyển sang PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Lưu vào buffer
            buffer = io.BytesIO()
            
            if format.upper() == 'JPEG':
                pil_image.save(buffer, format='JPEG', quality=quality)
            elif format.upper() == 'PNG':
                pil_image.save(buffer, format='PNG')
            else:
                raise ValueError(f"Định dạng không được hỗ trợ: {format}")
            
            # Chuyển thành base64
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            logger.debug(f"Chuyển ảnh thành base64: {len(image_base64)} chars")
            return image_base64
            
        except Exception as e:
            logger.error(f"Lỗi khi chuyển ảnh thành base64: {e}")
            raise
    
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """
        Chuyển base64 string thành ảnh
        
        Args:
            base64_string: Base64 string
            
        Returns:
            np.ndarray: Ảnh dạng numpy array
        """
        try:
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Chuyển thành numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decode ảnh
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Không thể decode base64 string thành ảnh")
            
            logger.debug(f"Chuyển base64 thành ảnh: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Lỗi khi chuyển base64 thành ảnh: {e}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa ảnh về khoảng [0, 1]
        
        Args:
            image: Ảnh input
            
        Returns:
            np.ndarray: Ảnh đã chuẩn hóa
        """
        try:
            # Chuyển về float32
            normalized = image.astype(np.float32)
            
            # Chuẩn hóa về khoảng [0, 1]
            normalized = normalized / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn hóa ảnh: {e}")
            raise
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Chuyển ảnh từ khoảng [0, 1] về [0, 255]
        
        Args:
            image: Ảnh đã chuẩn hóa
            
        Returns:
            np.ndarray: Ảnh đã denormalize
        """
        try:
            # Chuyển về khoảng [0, 255]
            denormalized = image * 255.0
            
            # Chuyển về uint8
            denormalized = np.clip(denormalized, 0, 255).astype(np.uint8)
            
            return denormalized
            
        except Exception as e:
            logger.error(f"Lỗi khi denormalize ảnh: {e}")
            raise
    
    def get_image_info(self, image: np.ndarray) -> Dict:
        """
        Lấy thông tin ảnh
        
        Args:
            image: Ảnh input
            
        Returns:
            Dict: Thông tin ảnh
        """
        try:
            h, w = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1
            dtype = str(image.dtype)
            
            info = {
                'width': w,
                'height': h,
                'channels': channels,
                'dtype': dtype,
                'size_bytes': image.nbytes,
                'shape': image.shape
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin ảnh: {e}")
            raise
    
    def create_image_grid(self, images: List[np.ndarray], grid_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Tạo grid ảnh
        
        Args:
            images: Danh sách ảnh
            grid_size: Kích thước grid (rows, cols). Nếu None sẽ tự động tính
            
        Returns:
            np.ndarray: Ảnh grid
        """
        try:
            if not images:
                raise ValueError("Danh sách ảnh không được rỗng")
            
            n_images = len(images)
            
            # Tự động tính grid size nếu không được cung cấp
            if grid_size is None:
                cols = int(np.ceil(np.sqrt(n_images)))
                rows = int(np.ceil(n_images / cols))
                grid_size = (rows, cols)
            
            rows, cols = grid_size
            
            # Lấy kích thước ảnh đầu tiên làm chuẩn
            target_size = (images[0].shape[1], images[0].shape[0])
            
            # Tạo grid
            grid_height = rows * target_size[1]
            grid_width = cols * target_size[0]
            
            if len(images[0].shape) == 3:
                grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            else:
                grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
            
            # Đặt ảnh vào grid
            for i, image in enumerate(images):
                if i >= rows * cols:
                    break
                
                row = i // cols
                col = i % cols
                
                # Resize ảnh về kích thước chuẩn
                resized = self.resize_image(image, target_size, keep_aspect_ratio=False)
                
                # Tính vị trí trong grid
                y_start = row * target_size[1]
                y_end = y_start + target_size[1]
                x_start = col * target_size[0]
                x_end = x_start + target_size[0]
                
                # Đặt ảnh vào grid
                grid[y_start:y_end, x_start:x_end] = resized
            
            logger.debug(f"Tạo grid ảnh: {grid_size} với {len(images)} ảnh")
            return grid
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo image grid: {e}")
            raise 