"""
Module xử lý khuôn mặt cho hệ thống nhận diện
Sử dụng DeepFace để trích xuất vector embedding và so khớp khuôn mặt
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json

from config.config import FACE_RECOGNITION_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

class FaceProcessor:
    """Lớp xử lý khuôn mặt với DeepFace"""
    
    def __init__(self):
        """Khởi tạo FaceProcessor với cấu hình từ config"""
        self.similarity_threshold = FACE_RECOGNITION_CONFIG['similarity_threshold']
        self.embedding_model = FACE_RECOGNITION_CONFIG['embedding_model']
        self.face_size = FACE_RECOGNITION_CONFIG['face_size']
        self.max_faces = FACE_RECOGNITION_CONFIG['max_faces']
        self.embeddings_dir = MODEL_CONFIG['face_embeddings_dir']
        
        # Tạo thư mục embeddings nếu chưa có
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache cho embeddings đã load
        self._embeddings_cache = {}
        self._metadata_cache = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Phát hiện khuôn mặt trong ảnh sử dụng OpenCV
        
        Args:
            image: Ảnh input dạng numpy array
            
        Returns:
            List[Dict]: Danh sách các khuôn mặt được phát hiện với bbox
        """
        # Chuyển sang grayscale cho face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sử dụng Haar Cascade để phát hiện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        detected_faces = []
        for (x, y, w, h) in faces:
            face_info = {
                'bbox': (x, y, w, h),
                'confidence': 1.0,  # Haar cascade không trả về confidence
                'face_region': image[y:y+h, x:x+w]
            }
            detected_faces.append(face_info)
        
        logger.info(f"Phát hiện {len(detected_faces)} khuôn mặt trong ảnh")
        return detected_faces
    
    def extract_face_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Trích xuất vector embedding từ khuôn mặt sử dụng DeepFace
        
        Args:
            face_image: Ảnh khuôn mặt đã được crop
            
        Returns:
            np.ndarray: Vector embedding 128 chiều
        """
        try:
            # Resize ảnh về kích thước chuẩn
            face_resized = cv2.resize(face_image, self.face_size)
            
            # Chuyển BGR sang RGB (DeepFace yêu cầu RGB)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Trích xuất embedding sử dụng DeepFace
            embedding = DeepFace.represent(
                img_path=face_rgb,
                model_name=self.embedding_model,
                enforce_detection=False
            )
            
            # DeepFace trả về list, lấy phần tử đầu tiên
            if isinstance(embedding, list) and len(embedding) > 0:
                embedding = embedding[0]['embedding']
            
            embedding = np.array(embedding, dtype=np.float32)
            
            logger.debug(f"Trích xuất embedding thành công: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất embedding: {e}")
            raise
    
    def save_face_embedding(self, embedding: np.ndarray, filename: str, employee_id: str):
        """
        Lưu vector embedding vào file
        
        Args:
            embedding: Vector embedding
            filename: Tên file ảnh gốc
            employee_id: ID nhân viên
        """
        try:
            # Tạo tên file embedding
            embedding_filename = f"{employee_id}_{Path(filename).stem}.npy"
            embedding_path = self.embeddings_dir / embedding_filename
            
            # Lưu embedding
            np.save(embedding_path, embedding)
            
            logger.info(f"Đã lưu embedding: {embedding_path}")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu embedding: {e}")
            raise
    
    def load_face_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load tất cả face embeddings từ thư mục
        
        Returns:
            Dict: Mapping employee_id -> embedding
        """
        if self._embeddings_cache:
            return self._embeddings_cache
        
        embeddings = {}
        embedding_files = list(self.embeddings_dir.glob("*.npy"))
        
        for embedding_file in embedding_files:
            try:
                # Lấy employee_id từ tên file
                employee_id = embedding_file.stem.split('_')[0]
                
                # Load embedding
                embedding = np.load(embedding_file)
                embeddings[employee_id] = embedding
                
            except Exception as e:
                logger.warning(f"Không thể load embedding từ {embedding_file}: {e}")
                continue
        
        self._embeddings_cache = embeddings
        logger.info(f"Đã load {len(embeddings)} embeddings")
        return embeddings
    
    def load_embeddings_metadata(self) -> pd.DataFrame:
        """
        Load metadata của embeddings
        
        Returns:
            pd.DataFrame: DataFrame chứa thông tin embeddings
        """
        if self._metadata_cache is not None:
            return self._metadata_cache
        
        metadata_file = self.embeddings_dir / 'embeddings_mapping.csv'
        
        if not metadata_file.exists():
            logger.warning("File embeddings mapping không tồn tại")
            return pd.DataFrame()
        
        try:
            metadata = pd.read_csv(metadata_file)
            self._metadata_cache = metadata
            logger.info(f"Đã load metadata với {len(metadata)} records")
            return metadata
            
        except Exception as e:
            logger.error(f"Lỗi khi load embeddings metadata: {e}")
            return pd.DataFrame()
    
    def find_most_similar_face(self, query_embedding: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Tìm khuôn mặt tương tự nhất với embedding query
        
        Args:
            query_embedding: Vector embedding của khuôn mặt cần tìm
            
        Returns:
            Tuple: (employee_id, similarity_score, metadata)
        """
        # Load embeddings và metadata
        embeddings = self.load_face_embeddings()
        metadata = self.load_embeddings_metadata()
        
        if not embeddings:
            logger.warning("Không có embeddings nào để so khớp")
            return None, 0.0, {}
        
        # Tính similarity với tất cả embeddings
        similarities = {}
        for employee_id, embedding in embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), 
                embedding.reshape(1, -1)
            )[0][0]
            similarities[employee_id] = similarity
        
        # Tìm employee_id có similarity cao nhất
        if not similarities:
            return None, 0.0, {}
        
        best_employee_id = max(similarities, key=similarities.get)
        best_similarity = similarities[best_employee_id]
        
        # Kiểm tra ngưỡng similarity
        if best_similarity < self.similarity_threshold:
            logger.info(f"Similarity {best_similarity:.3f} thấp hơn ngưỡng {self.similarity_threshold}")
            return None, best_similarity, {}
        
        # Lấy metadata của employee
        employee_metadata = {}
        if not metadata.empty:
            employee_data = metadata[metadata['employee_id'] == best_employee_id]
            if not employee_data.empty:
                employee_metadata = employee_data.iloc[0].to_dict()
        
        logger.info(f"Tìm thấy match: {best_employee_id} với similarity {best_similarity:.3f}")
        return best_employee_id, best_similarity, employee_metadata
    
    def process_image_for_recognition(self, image: np.ndarray) -> List[Dict]:
        """
        Xử lý ảnh để nhận diện khuôn mặt
        
        Args:
            image: Ảnh input
            
        Returns:
            List[Dict]: Kết quả nhận diện cho từng khuôn mặt
        """
        # Phát hiện khuôn mặt
        detected_faces = self.detect_faces(image)
        
        if not detected_faces:
            logger.info("Không phát hiện khuôn mặt nào trong ảnh")
            return []
        
        recognition_results = []
        
        for i, face_info in enumerate(detected_faces):
            try:
                # Trích xuất embedding
                face_embedding = self.extract_face_embedding(face_info['face_region'])
                
                # Tìm khuôn mặt tương tự
                employee_id, similarity, metadata = self.find_most_similar_face(face_embedding)
                
                result = {
                    'face_index': i,
                    'bbox': face_info['bbox'],
                    'confidence': face_info['confidence'],
                    'employee_id': employee_id,
                    'full_name': metadata.get('full_name', 'Unknown') if employee_id else 'Unknown',
                    'similarity_score': similarity,
                    'is_recognized': employee_id is not None
                }
                
                recognition_results.append(result)
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý khuôn mặt {i}: {e}")
                continue
        
        return recognition_results
    
    def create_embeddings_from_dataset(self, metadata_file: Path, images_dir: Path):
        """
        Tạo embeddings cho toàn bộ dataset
        
        Args:
            metadata_file: Đường dẫn đến file metadata
            images_dir: Thư mục chứa ảnh
        """
        try:
            metadata = pd.read_csv(metadata_file)
            total_images = len(metadata)
            processed_count = 0
            
            logger.info(f"Bắt đầu tạo embeddings cho {total_images} ảnh")
            
            for idx, row in metadata.iterrows():
                try:
                    filename = row['filename']
                    employee_id = row['employee_id']
                    
                    # Đường dẫn ảnh
                    image_path = images_dir / filename
                    
                    if not image_path.exists():
                        logger.warning(f"Ảnh không tồn tại: {image_path}")
                        continue
                    
                    # Đọc ảnh
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Không thể đọc ảnh: {image_path}")
                        continue
                    
                    # Phát hiện khuôn mặt
                    detected_faces = self.detect_faces(image)
                    
                    if not detected_faces:
                        logger.warning(f"Không phát hiện khuôn mặt trong: {filename}")
                        continue
                    
                    # Lấy khuôn mặt đầu tiên (giả sử mỗi ảnh chỉ có 1 khuôn mặt)
                    face_info = detected_faces[0]
                    
                    # Trích xuất embedding
                    embedding = self.extract_face_embedding(face_info['face_region'])
                    
                    # Lưu embedding
                    self.save_face_embedding(embedding, filename, employee_id)
                    
                    processed_count += 1
                    
                    if (idx + 1) % 10 == 0:
                        logger.info(f"Đã xử lý {idx + 1}/{total_images} ảnh")
                
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý ảnh {filename}: {e}")
                    continue
            
            logger.info(f"Hoàn thành tạo embeddings: {processed_count}/{total_images} ảnh")
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings từ dataset: {e}")
            raise
    
    def get_embedding_statistics(self) -> Dict:
        """Lấy thống kê về embeddings"""
        embeddings = self.load_face_embeddings()
        metadata = self.load_embeddings_metadata()
        
        stats = {
            'total_embeddings': len(embeddings),
            'total_metadata_records': len(metadata) if not metadata.empty else 0,
            'unique_employees': len(set(embeddings.keys())),
            'embedding_dimension': list(embeddings.values())[0].shape[0] if embeddings else 0
        }
        
        return stats 