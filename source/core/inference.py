"""
Inference script cho hệ thống nhận diện khuôn mặt
Sử dụng YOLOv7 cho face detection và DeepFace cho face recognition
"""

import os
import sys
import logging
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Thêm đường dẫn để import các module
sys.path.append(str(Path(__file__).parent.parent))

from utils.face_utils import FaceProcessor
from utils.image_utils import ImageProcessor
from config.config import INFERENCE_CONFIG, MODEL_CONFIG, FACE_RECOGNITION_CONFIG

# Thiết lập logging
logging.basicConfig(
    level=getattr(logging, INFERENCE_CONFIG['logging']['level']),
    format=INFERENCE_CONFIG['logging']['format'],
    handlers=[
        logging.FileHandler(INFERENCE_CONFIG['logging']['file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FaceRecognitionInference:
    """
    Lớp inference cho hệ thống nhận diện khuôn mặt
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Khởi tạo inference engine
        
        Args:
            model_path: Đường dẫn đến model weights (optional)
        """
        self.face_processor = FaceProcessor()
        self.image_processor = ImageProcessor()
        
        # Đường dẫn
        self.models_dir = Path(MODEL_CONFIG['models_dir'])
        self.embeddings_dir = self.models_dir / 'face_embeddings'
        self.output_dir = Path(INFERENCE_CONFIG['output_dir'])
        
        # Tạo thư mục output nếu chưa có
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model path
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self.models_dir / 'trained' / 'face_detection' / 'weights' / 'best.pt'
        
        # Kiểm tra model
        if not self.model_path.exists():
            logger.warning(f"Không tìm thấy model tại: {self.model_path}")
            logger.info("Sẽ sử dụng OpenCV face detection")
        
        logger.info("Khởi tạo Face Recognition Inference")
        logger.info(f"Model path: {self.model_path}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image: Ảnh input
            
        Returns:
            List[Dict]: Danh sách khuôn mặt được phát hiện
        """
        try:
            # Sử dụng OpenCV face detection (đơn giản và nhanh)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detected_faces = []
            for (x, y, w, h) in faces:
                detected_faces.append({
                    'bbox': [x, y, w, h],
                    'confidence': 0.9,
                    'class_name': 'face'
                })
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"Lỗi khi phát hiện khuôn mặt: {e}")
            return []
    
    def recognize_faces(self, image: np.ndarray, detected_faces: List[Dict]) -> List[Dict]:
        """
        Nhận diện khuôn mặt
        
        Args:
            image: Ảnh input
            detected_faces: Danh sách khuôn mặt đã phát hiện
            
        Returns:
            List[Dict]: Kết quả nhận diện
        """
        try:
            recognition_results = []
            
            for i, face_info in enumerate(detected_faces):
                try:
                    # Cắt khuôn mặt
                    x, y, w, h = face_info['bbox']
                    face_region = image[y:y+h, x:x+w]
                    
                    # Trích xuất embedding
                    face_embedding = self.face_processor.extract_face_embedding(face_region)
                    
                    if face_embedding is not None:
                        # Tìm khuôn mặt tương tự
                        employee_id, similarity, metadata = self.face_processor.find_most_similar_face(face_embedding)
                        
                        result = {
                            'face_index': i,
                            'bbox': face_info['bbox'],
                            'confidence': face_info['confidence'],
                            'employee_id': employee_id,
                            'full_name': metadata.get('full_name', 'Unknown') if metadata else 'Unknown',
                            'similarity': similarity,
                            'is_recognized': similarity >= FACE_RECOGNITION_CONFIG['similarity_threshold']
                        }
                        
                        recognition_results.append(result)
                        
                        logger.info(f"Face {i}: {result['full_name']} (ID: {result['employee_id']}, Similarity: {similarity:.3f})")
                    else:
                        logger.warning(f"Không thể trích xuất embedding cho face {i}")
                        
                except Exception as e:
                    logger.error(f"Lỗi khi nhận diện face {i}: {e}")
                    continue
            
            return recognition_results
            
        except Exception as e:
            logger.error(f"Lỗi khi nhận diện khuôn mặt: {e}")
            return []
    
    def process_single_image(self, image_path: str, save_result: bool = True) -> Dict:
        """
        Xử lý một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            save_result: Có lưu kết quả không
            
        Returns:
            Dict: Kết quả xử lý
        """
        try:
            start_time = time.time()
            
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'error': 'Không thể đọc ảnh',
                    'status': 'error'
                }
            
            # Phát hiện khuôn mặt
            detected_faces = self.detect_faces(image)
            
            if len(detected_faces) == 0:
                return {
                    'image_path': image_path,
                    'faces_detected': 0,
                    'faces_recognized': 0,
                    'processing_time': time.time() - start_time,
                    'status': 'no_faces_detected'
                }
            
            # Nhận diện khuôn mặt
            recognition_results = self.recognize_faces(image, detected_faces)
            
            # Vẽ kết quả lên ảnh
            result_image = image.copy()
            for result in recognition_results:
                x, y, w, h = result['bbox']
                
                # Màu sắc dựa trên kết quả nhận diện
                if result['is_recognized']:
                    color = (0, 255, 0)  # Xanh lá
                    label = f"{result['full_name']} ({result['similarity']:.2f})"
                else:
                    color = (0, 0, 255)  # Đỏ
                    label = "Unknown"
                
                # Vẽ bounding box
                cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
                
                # Vẽ label
                cv2.putText(result_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Lưu kết quả
            if save_result:
                output_path = self.output_dir / f"result_{Path(image_path).stem}.jpg"
                cv2.imwrite(str(output_path), result_image)
                logger.info(f"Lưu kết quả tại: {output_path}")
            
            processing_time = time.time() - start_time
            
            return {
                'image_path': image_path,
                'faces_detected': len(detected_faces),
                'faces_recognized': sum(1 for r in recognition_results if r['is_recognized']),
                'recognition_results': recognition_results,
                'processing_time': processing_time,
                'output_path': str(output_path) if save_result else None,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'status': 'error'
            }
    
    def process_batch_images(self, image_dir: str) -> Dict:
        """
        Xử lý batch ảnh
        
        Args:
            image_dir: Thư mục chứa ảnh
            
        Returns:
            Dict: Kết quả xử lý batch
        """
        try:
            logger.info(f"Bắt đầu xử lý batch từ thư mục: {image_dir}")
            
            # Tìm tất cả ảnh
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(image_dir).glob(f"*{ext}"))
                image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
            
            if not image_files:
                return {
                    'error': 'Không tìm thấy ảnh nào',
                    'status': 'error'
                }
            
            logger.info(f"Tìm thấy {len(image_files)} ảnh")
            
            # Xử lý từng ảnh
            results = []
            total_faces = 0
            total_recognized = 0
            
            for image_file in image_files:
                result = self.process_single_image(str(image_file))
                results.append(result)
                
                if result['status'] == 'success':
                    total_faces += result['faces_detected']
                    total_recognized += result['faces_recognized']
            
            # Tạo báo cáo
            report = {
                'total_images': len(image_files),
                'processed_images': len([r for r in results if r['status'] == 'success']),
                'total_faces_detected': total_faces,
                'total_faces_recognized': total_recognized,
                'recognition_rate': total_recognized / total_faces if total_faces > 0 else 0,
                'results': results,
                'status': 'success'
            }
            
            # Lưu báo cáo
            report_file = self.output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Hoàn thành batch processing. Báo cáo: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý batch: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def run_realtime_detection(self, camera_id: int = 0):
        """
        Chạy nhận diện realtime từ camera
        
        Args:
            camera_id: ID camera (mặc định 0)
        """
        try:
            logger.info(f"Bắt đầu realtime detection từ camera {camera_id}")
            
            # Mở camera
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"Không thể mở camera {camera_id}")
                return
            
            logger.info("Nhấn 'q' để thoát, 's' để lưu ảnh")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Không thể đọc frame từ camera")
                    break
                
                # Phát hiện khuôn mặt
                detected_faces = self.detect_faces(frame)
                
                # Nhận diện khuôn mặt
                recognition_results = self.recognize_faces(frame, detected_faces)
                
                # Vẽ kết quả
                for result in recognition_results:
                    x, y, w, h = result['bbox']
                    
                    if result['is_recognized']:
                        color = (0, 255, 0)
                        label = f"{result['full_name']} ({result['similarity']:.2f})"
                    else:
                        color = (0, 0, 255)
                        label = "Unknown"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Hiển thị thông tin
                cv2.putText(frame, f"Faces: {len(detected_faces)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Hiển thị frame
                cv2.imshow('Face Recognition', frame)
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Lưu ảnh
                    save_path = self.output_dir / f"realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(save_path), frame)
                    logger.info(f"Lưu ảnh tại: {save_path}")
            
            # Giải phóng tài nguyên
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Lỗi trong realtime detection: {e}")

def main():
    """Hàm main để chạy inference"""
    parser = argparse.ArgumentParser(description='Face Recognition Inference')
    parser.add_argument('--mode', choices=['single', 'batch', 'realtime'], 
                       default='single', help='Chế độ chạy')
    parser.add_argument('--input', type=str, help='Đường dẫn ảnh hoặc thư mục')
    parser.add_argument('--model', type=str, help='Đường dẫn model weights')
    parser.add_argument('--camera', type=int, default=0, help='ID camera cho realtime')
    
    args = parser.parse_args()
    
    # Khởi tạo inference engine
    inference = FaceRecognitionInference(model_path=args.model)
    
    if args.mode == 'single':
        if not args.input:
            print("Cần cung cấp đường dẫn ảnh với --input")
            return
        
        result = inference.process_single_image(args.input)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.mode == 'batch':
        if not args.input:
            print("Cần cung cấp thư mục ảnh với --input")
            return
        
        result = inference.process_batch_images(args.input)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.mode == 'realtime':
        inference.run_realtime_detection(args.camera)

if __name__ == '__main__':
    main() 