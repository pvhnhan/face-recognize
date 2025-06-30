"""
API endpoints cho hệ thống nhận diện khuôn mặt
Sử dụng YOLOv7 chính thức từ repository GitHub: https://github.com/WongKinYiu/yolov7

Các endpoints:
- /detect: Phát hiện khuôn mặt bằng YOLOv7
- /recognize: Nhận diện khuôn mặt (detection + recognition)
- /batch: Xử lý batch ảnh
- /status: Trạng thái hệ thống
"""

import os
import sys
import logging
import base64
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Resource, fields, Namespace
import cv2
import numpy as np

# Thêm đường dẫn để import các module
sys.path.append(str(Path(__file__).parent.parent))

from utils.face_utils import FaceProcessor
from utils.image_utils import ImageProcessor
from config.config import FACE_RECOGNITION_CONFIG, MODEL_CONFIG, FLASK_CONFIG

# Thiết lập logging
logger = logging.getLogger(__name__)

# Tạo Blueprint
face_recognition_bp = Blueprint('face_recognition', __name__, url_prefix='/api/face-recognition')

# Tạo namespace cho Swagger
ns = Namespace('face-recognition', description='Face recognition operations')

# Định nghĩa models cho Swagger
face_detection_model = ns.model('FaceDetection', {
    'faces': fields.List(fields.Raw, description='Danh sách khuôn mặt được phát hiện'),
    'total_faces': fields.Integer(description='Tổng số khuôn mặt'),
    'status': fields.String(description='Trạng thái xử lý')
})

face_recognition_model = ns.model('FaceRecognition', {
    'full_name': fields.String(description='Tên đầy đủ của người được nhận diện'),
    'employee_id': fields.String(description='ID nhân viên'),
    'input_image': fields.String(description='Ảnh input dạng base64'),
    'matched_images': fields.List(fields.String, description='Danh sách ảnh khớp'),
    'similarity_score': fields.Float(description='Độ tương đồng (0-1)'),
    'status': fields.String(description='Trạng thái xử lý')
})

batch_recognition_model = ns.model('BatchRecognition', {
    'results': fields.List(fields.Raw, description='Kết quả nhận diện cho từng ảnh'),
    'total_processed': fields.Integer(description='Tổng số ảnh đã xử lý'),
    'status': fields.String(description='Trạng thái xử lý')
})

error_model = ns.model('Error', {
    'error': fields.String(description='Mô tả lỗi'),
    'status': fields.String(description='Trạng thái lỗi')
})

class FaceRecognitionAPI:
    """
    Lớp xử lý API nhận diện khuôn mặt
    Sử dụng YOLOv7 chính thức cho face detection và DeepFace cho recognition
    """
    
    def __init__(self):
        """Khởi tạo API"""
        self.face_processor = FaceProcessor()
        self.image_processor = ImageProcessor()
        
        # Kiểm tra YOLOv7 repository
        self.yolov7_path = Path(__file__).parent.parent / 'yolov7'
        self.yolov7_available = self._check_yolov7_availability()
        
        # Weights path
        self.weights_path = str(MODEL_CONFIG['trained_weights'])
        
        logger.info("Khởi tạo Face Recognition API")
        logger.info(f"YOLOv7 available: {self.yolov7_available}")
    
    def _check_yolov7_availability(self) -> bool:
        """
        Kiểm tra YOLOv7 repository có sẵn không
        
        Returns:
            bool: True nếu YOLOv7 có sẵn
        """
        if not self.yolov7_path.exists():
            logger.warning("YOLOv7 repository không tồn tại")
            return False
        
        # Kiểm tra các file cần thiết
        required_files = [
            'detect.py',
            'models/experimental.py',
            'utils/general.py',
            'utils/augmentations.py'
        ]
        
        for file_path in required_files:
            if not (self.yolov7_path / file_path).exists():
                logger.warning(f"Thiếu file YOLOv7: {file_path}")
                return False
        
        logger.info("YOLOv7 repository đã sẵn sàng cho API")
        return True
    
    def detect_faces_yolov7(self, image_path: str) -> List[Dict]:
        """
        Phát hiện khuôn mặt sử dụng YOLOv7 chính thức
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            List[Dict]: Danh sách khuôn mặt được phát hiện
        """
        if not self.yolov7_available:
            logger.warning("YOLOv7 không khả dụng, sử dụng OpenCV fallback")
            return self._detect_faces_opencv(image_path)
        
        logger.info(f"Phát hiện khuôn mặt bằng YOLOv7: {image_path}")
        
        try:
            # Tạo lệnh inference theo YOLOv7 chính thức
            inference_cmd = self._prepare_yolov7_inference_command(image_path)
            
            # Chạy inference trong thư mục YOLOv7
            result = subprocess.run(
                inference_cmd,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.yolov7_path
            )
            
            if result.returncode == 0:
                logger.info("YOLOv7 inference thành công!")
                # Parse kết quả từ output
                detected_faces = self._parse_yolov7_output(result.stdout, image_path)
                return detected_faces
            else:
                logger.error("Lỗi trong YOLOv7 inference:")
                logger.error(result.stderr)
                return self._detect_faces_opencv(image_path)
                
        except Exception as e:
            logger.error(f"Lỗi khi chạy YOLOv7 inference: {e}")
            return self._detect_faces_opencv(image_path)
    
    def _prepare_yolov7_inference_command(self, image_path: str) -> str:
        """
        Tạo lệnh inference theo format YOLOv7 chính thức
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            str: Lệnh inference
        """
        # Theo YOLOv7 chính thức: https://github.com/WongKinYiu/yolov7
        cmd_parts = [
            "python detect.py",
            f"--weights {self.weights_path}",
            f"--source {image_path}",
            f"--conf {MODEL_CONFIG['confidence_threshold']}",
            f"--iou {MODEL_CONFIG['nms_threshold']}",
            f"--img-size {MODEL_CONFIG.get('img_size', 640)}",
            "--device 0" if torch.cuda.is_available() else "--device cpu",
            "--project ../outputs",
            "--name face_detection_results",
            "--exist-ok",
            "--save-txt"  # Lưu kết quả dạng text
        ]
        
        return " ".join(cmd_parts)
    
    def _parse_yolov7_output(self, output: str, image_path: str) -> List[Dict]:
        """
        Parse kết quả từ YOLOv7 output
        
        Args:
            output: Output từ YOLOv7
            image_path: Đường dẫn đến ảnh gốc
            
        Returns:
            List[Dict]: Danh sách khuôn mặt được phát hiện
        """
        detected_faces = []
        
        try:
            # Đọc ảnh để lấy kích thước
            image = cv2.imread(image_path)
            if image is None:
                return detected_faces
            
            h, w = image.shape[:2]
            
            # Tìm file labels trong output
            output_dir = Path('outputs/face_detection_results/labels')
            image_name = Path(image_path).stem
            
            label_file = output_dir / f"{image_name}.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Format YOLO: class x_center y_center width height confidence
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        width = float(parts[3]) * w
                        height = float(parts[4]) * h
                        confidence = float(parts[5]) if len(parts) > 5 else 0.9
                        
                        # Chuyển đổi sang format bbox
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        face_info = {
                            'bbox': (x1, y1, x2 - x1, y2 - y1),
                            'confidence': confidence,
                            'face_region': image[y1:y2, x1:x2],
                            'class_id': class_id,
                            'class_name': 'face'
                        }
                        detected_faces.append(face_info)
            
            logger.info(f"YOLOv7 phát hiện {len(detected_faces)} khuôn mặt")
            
        except Exception as e:
            logger.error(f"Lỗi khi parse YOLOv7 output: {e}")
        
        return detected_faces
    
    def _detect_faces_opencv(self, image_path: str) -> List[Dict]:
        """
        Fallback face detection sử dụng OpenCV
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            List[Dict]: Danh sách khuôn mặt được phát hiện
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            # Sử dụng OpenCV face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detected_faces = []
            for (x, y, w, h) in faces:
                face_info = {
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,  # Giá trị mặc định
                    'face_region': image[y:y+h, x:x+w],
                    'class_id': 0,
                    'class_name': 'face'
                }
                detected_faces.append(face_info)
            
            logger.info(f"OpenCV fallback phát hiện {len(detected_faces)} khuôn mặt")
            return detected_faces
            
        except Exception as e:
            logger.error(f"Lỗi trong OpenCV face detection: {e}")
            return []
    
    def recognize_faces(self, image_path: str) -> List[Dict]:
        """
        Nhận diện khuôn mặt trong ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            
        Returns:
            List[Dict]: Kết quả nhận diện cho từng khuôn mặt
        """
        logger.info(f"Bắt đầu nhận diện khuôn mặt: {image_path}")
        
        # Phát hiện khuôn mặt
        detected_faces = self.detect_faces_yolov7(image_path)
        
        if not detected_faces:
            return []
        
        recognition_results = []
        
        for i, face_info in enumerate(detected_faces):
            try:
                # Trích xuất embedding
                face_embedding = self.face_processor.extract_face_embedding(face_info['face_region'])
                
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
                
            except Exception as e:
                logger.error(f"Lỗi khi nhận diện face {i}: {e}")
                result = {
                    'face_index': i,
                    'bbox': face_info['bbox'],
                    'confidence': face_info['confidence'],
                    'employee_id': 'Unknown',
                    'full_name': 'Unknown',
                    'similarity': 0.0,
                    'is_recognized': False,
                    'error': str(e)
                }
                recognition_results.append(result)
        
        return recognition_results

# Khởi tạo API instance
api_instance = FaceRecognitionAPI()

# Flask-RESTX Resource Classes cho Swagger
@ns.route('/detect')
class FaceDetectionAPI(Resource):
    @ns.doc('detect_faces')
    @ns.response(200, 'Success', face_detection_model)
    @ns.response(400, 'Bad Request', error_model)
    @ns.response(500, 'Internal Server Error', error_model)
    def post(self):
        """
        Phát hiện khuôn mặt trong ảnh
        
        Gửi ảnh dưới dạng file upload hoặc base64 string
        """
        try:
            start_time = time.time()
            
            # Kiểm tra request
            if 'image' not in request.files and 'image_base64' not in request.form:
                return {
                    'error': 'Thiếu ảnh input',
                    'message': 'Gửi file ảnh hoặc base64 string',
                    'status': 'error'
                }, 400
            
            # Xử lý ảnh input
            image_path = None
            if 'image' in request.files:
                # Upload file
                file = request.files['image']
                if file.filename == '':
                    return {'error': 'Không có file được chọn', 'status': 'error'}, 400
                
                # Kiểm tra định dạng file
                allowed_extensions = FLASK_CONFIG['allowed_extensions']
                if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
                    return {
                        'error': 'Định dạng file không được hỗ trợ',
                        'allowed_formats': allowed_extensions,
                        'status': 'error'
                    }, 400
                
                # Lưu file tạm
                temp_dir = FLASK_CONFIG['temp_dir']
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                image_path = temp_dir / f"temp_{int(time.time())}_{file.filename}"
                file.save(str(image_path))
                
            elif 'image_base64' in request.form:
                # Base64 string
                image_base64 = request.form['image_base64']
                try:
                    # Lưu base64 thành file tạm
                    temp_dir = FLASK_CONFIG['temp_dir']
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    image_path = temp_dir / f"temp_{int(time.time())}.jpg"
                    image = api_instance.image_processor.base64_to_image(image_base64)
                    cv2.imwrite(str(image_path), image)
                    
                except Exception as e:
                    return {'error': f'Lỗi khi decode base64: {str(e)}', 'status': 'error'}, 400
            
            if image_path is None:
                return {'error': 'Không thể xử lý ảnh input', 'status': 'error'}, 400
            
            try:
                # Phát hiện khuôn mặt
                detected_faces = api_instance.detect_faces_yolov7(str(image_path))
                
                # Đọc ảnh để vẽ kết quả
                image = cv2.imread(str(image_path))
                if image is not None:
                    # Vẽ kết quả lên ảnh
                    result_image = api_instance.image_processor.draw_detection_results(image, detected_faces)
                    
                    # Chuyển đổi ảnh sang base64
                    result_image_base64 = api_instance.image_processor.image_to_base64(result_image)
                else:
                    result_image_base64 = ""
                
                # Chuẩn bị response
                processing_time = time.time() - start_time
                
                return {
                    'faces': detected_faces,
                    'total_faces': len(detected_faces),
                    'result_image_base64': result_image_base64,
                    'processing_time': processing_time,
                    'status': 'success'
                }
                
            finally:
                # Xóa file tạm
                if image_path and image_path.exists():
                    try:
                        image_path.unlink()
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Lỗi trong API detect_faces: {e}")
            return {
                'error': 'Lỗi server',
                'message': str(e),
                'status': 'error'
            }, 500

@ns.route('/recognize')
class FaceRecognitionAPI(Resource):
    @ns.doc('recognize_faces')
    @ns.response(200, 'Success', face_recognition_model)
    @ns.response(400, 'Bad Request', error_model)
    @ns.response(500, 'Internal Server Error', error_model)
    def post(self):
        """
        Nhận diện khuôn mặt trong ảnh
        
        Gửi ảnh dưới dạng file upload hoặc base64 string
        """
        try:
            start_time = time.time()
            
            # Kiểm tra request
            if 'image' not in request.files and 'image_base64' not in request.form:
                return {
                    'error': 'Thiếu ảnh input',
                    'message': 'Gửi file ảnh hoặc base64 string',
                    'status': 'error'
                }, 400
            
            # Xử lý ảnh input (tương tự như detect)
            image_path = None
            if 'image' in request.files:
                file = request.files['image']
                if file.filename == '':
                    return {'error': 'Không có file được chọn', 'status': 'error'}, 400
                
                temp_dir = FLASK_CONFIG['temp_dir']
                temp_dir.mkdir(parents=True, exist_ok=True)
                image_path = temp_dir / f"temp_{int(time.time())}_{file.filename}"
                file.save(str(image_path))
                
            elif 'image_base64' in request.form:
                image_base64 = request.form['image_base64']
                try:
                    temp_dir = FLASK_CONFIG['temp_dir']
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    image_path = temp_dir / f"temp_{int(time.time())}.jpg"
                    image = api_instance.image_processor.base64_to_image(image_base64)
                    cv2.imwrite(str(image_path), image)
                except Exception as e:
                    return {'error': f'Lỗi khi decode base64: {str(e)}', 'status': 'error'}, 400
            
            if image_path is None:
                return {'error': 'Không thể xử lý ảnh input', 'status': 'error'}, 400
            
            try:
                # Nhận diện khuôn mặt
                recognition_results = api_instance.recognize_faces(str(image_path))
                
                if not recognition_results:
                    return {
                        'full_name': 'Unknown',
                        'employee_id': None,
                        'input_image': api_instance.image_processor.image_to_base64(cv2.imread(str(image_path))),
                        'matched_images': [],
                        'similarity_score': 0.0,
                        'status': 'no_face_detected'
                    }
                
                # Lấy kết quả đầu tiên
                result = recognition_results[0]
                
                processing_time = time.time() - start_time
                
                return {
                    'full_name': result['full_name'],
                    'employee_id': result['employee_id'],
                    'input_image': api_instance.image_processor.image_to_base64(cv2.imread(str(image_path))),
                    'matched_images': [result.get('matched_image', '')],
                    'similarity_score': result['similarity'],
                    'status': 'success'
                }
                
            finally:
                # Xóa file tạm
                if image_path and image_path.exists():
                    try:
                        image_path.unlink()
                    except:
                        pass
        
        except Exception as e:
            logger.error(f"Lỗi trong API recognize_faces: {e}")
            return {
                'error': 'Lỗi server',
                'message': str(e),
                'status': 'error'
            }, 500

@ns.route('/batch')
class BatchRecognitionAPI(Resource):
    @ns.doc('batch_recognition')
    @ns.response(200, 'Success', batch_recognition_model)
    @ns.response(400, 'Bad Request', error_model)
    @ns.response(500, 'Internal Server Error', error_model)
    def post(self):
        """
        Nhận diện khuôn mặt cho nhiều ảnh
        
        Gửi danh sách ảnh dưới dạng base64 strings
        """
        try:
            start_time = time.time()
            
            # Kiểm tra request
            if 'images' not in request.json:
                return {
                    'error': 'Thiếu danh sách ảnh',
                    'message': 'Gửi danh sách ảnh dưới dạng base64',
                    'status': 'error'
                }, 400
            
            images_base64 = request.json['images']
            if not isinstance(images_base64, list):
                return {
                    'error': 'Images phải là danh sách',
                    'status': 'error'
                }, 400
            
            results = []
            for i, image_base64 in enumerate(images_base64):
                try:
                    # Decode base64
                    image = api_instance.image_processor.base64_to_image(image_base64)
                    
                    # Lưu tạm để xử lý
                    temp_dir = FLASK_CONFIG['temp_dir']
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = temp_dir / f"temp_batch_{i}_{int(time.time())}.jpg"
                    cv2.imwrite(str(temp_path), image)
                    
                    # Nhận diện
                    recognition_results = api_instance.recognize_faces(str(temp_path))
                    
                    if recognition_results:
                        result = recognition_results[0]
                        results.append({
                            'image_index': i,
                            'full_name': result['full_name'],
                            'employee_id': result['employee_id'],
                            'similarity_score': result['similarity'],
                            'status': 'success'
                        })
                    else:
                        results.append({
                            'image_index': i,
                            'full_name': 'Unknown',
                            'employee_id': None,
                            'similarity_score': 0.0,
                            'status': 'no_face_detected'
                        })
                    
                    # Xóa file tạm
                    if temp_path.exists():
                        temp_path.unlink()
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý ảnh {i}: {e}")
                    results.append({
                        'image_index': i,
                        'full_name': 'Unknown',
                        'employee_id': None,
                        'similarity_score': 0.0,
                        'status': 'error',
                        'error': str(e)
                    })
            
            processing_time = time.time() - start_time
            
            return {
                'results': results,
                'total_processed': len(images_base64),
                'processing_time': processing_time,
                'status': 'success'
            }
        
        except Exception as e:
            logger.error(f"Lỗi trong API batch_recognition: {e}")
            return {
                'error': 'Lỗi server',
                'message': str(e),
                'status': 'error'
            }, 500

@ns.route('/status')
class StatusAPI(Resource):
    @ns.doc('get_status')
    @ns.response(200, 'Success')
    def get(self):
        """
        Lấy trạng thái hệ thống
        """
        try:
            # Lấy thống kê embeddings
            embeddings_stats = api_instance.face_processor.get_embedding_statistics()
            
            return {
                'status': 'running',
                'embeddings': embeddings_stats,
                'message': 'Hệ thống hoạt động bình thường'
            }
        
        except Exception as e:
            logger.error(f"Lỗi khi lấy status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }, 500

# Training status tracking
train_status = {'running': False, 'success': None, 'log': '', 'start_time': None}

@ns.route('/train')
class FaceTrainAPI(Resource):
    @ns.doc('start_training')
    @ns.response(200, 'Training started')
    @ns.response(409, 'Training already running')
    @ns.response(500, 'Internal Server Error', error_model)
    def post(self):
        """
        Bắt đầu quá trình training model
        """
        try:
            if train_status['running']:
                return {
                    'status': 'training', 
                    'message': 'Training is already running.',
                    'start_time': train_status['start_time']
                }, 409
            
            def train_job():
                """Background training job"""
                try:
                    train_status['running'] = True
                    train_status['start_time'] = time.time()
                    train_status['log'] = 'Bắt đầu training...'
                    
                    # Import và khởi tạo trainer
                    from core.train import FaceRecognitionTrainer
                    
                    # Khởi tạo trainer
                    trainer = FaceRecognitionTrainer()
                    train_status['log'] = 'Đã khởi tạo trainer, bắt đầu pipeline...'
                    
                    # Chạy pipeline training
                    result = trainer.run_training_pipeline()
                    
                    train_status['success'] = result
                    if result:
                        train_status['log'] = 'Training hoàn thành thành công!'
                    else:
                        train_status['log'] = 'Training thất bại!'
                    
                except Exception as e:
                    train_status['success'] = False
                    train_status['log'] = f'Lỗi training: {str(e)}'
                    logger.error(f"Lỗi trong training job: {e}")
                finally:
                    train_status['running'] = False
            
            # Chạy training trong thread riêng
            from threading import Thread
            Thread(target=train_job, daemon=True).start()
            
            return {
                'status': 'started',
                'message': 'Training đã được bắt đầu',
                'start_time': train_status['start_time']
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi bắt đầu training: {e}")
            return {
                'error': 'Lỗi server',
                'message': str(e),
                'status': 'error'
            }, 500

@ns.route('/train/status')
class FaceTrainStatusAPI(Resource):
    @ns.doc('get_training_status')
    @ns.response(200, 'Training status')
    def get(self):
        """
        Lấy trạng thái training
        """
        try:
            status_info = train_status.copy()
            
            # Tính thời gian đã chạy
            if status_info['running'] and status_info['start_time']:
                elapsed_time = time.time() - status_info['start_time']
                status_info['elapsed_time'] = elapsed_time
            else:
                status_info['elapsed_time'] = 0
            
            return status_info
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy training status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }, 500 