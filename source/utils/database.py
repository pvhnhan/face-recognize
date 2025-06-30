"""
Module quản lý database MongoDB cho hệ thống nhận diện khuôn mặt
Sử dụng cấu trúc tối ưu: config.py cho cấu hình hệ thống, .env cho biến môi trường
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    print("⚠️  PyMongo không được cài đặt. Chạy: pip install pymongo")

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBManager:
    """Quản lý kết nối và thao tác với MongoDB"""
    
    def __init__(self):
        """Khởi tạo MongoDB manager với cấu hình từ biến môi trường"""
        
        # Load cấu hình từ biến môi trường
        self.mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/face_recognition')
        self.mongodb_host = os.getenv('MONGODB_HOST', 'localhost')
        self.mongodb_port = int(os.getenv('MONGODB_PORT', 27017))
        self.mongodb_database = os.getenv('MONGODB_DATABASE', 'face_recognition')
        self.mongodb_username = os.getenv('MONGODB_USERNAME', '')
        self.mongodb_password = os.getenv('MONGODB_PASSWORD', '')
        
        # Collection names từ biến môi trường
        self.collections = {
            'employees': os.getenv('MONGODB_COLLECTION_EMPLOYEES', 'employees'),
            'face_embeddings': os.getenv('MONGODB_COLLECTION_FACE_EMBEDDINGS', 'face_embeddings'),
            'recognition_logs': os.getenv('MONGODB_COLLECTION_RECOGNITION_LOGS', 'recognition_logs'),
            'training_logs': os.getenv('MONGODB_COLLECTION_TRAINING_LOGS', 'training_logs'),
        }
        
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self) -> bool:
        """Kết nối đến MongoDB"""
        if not MONGODB_AVAILABLE:
            logger.error("PyMongo không khả dụng")
            return False
        
        try:
            # Tạo connection string
            if self.mongodb_username and self.mongodb_password:
                # Với authentication
                connection_string = f"mongodb://{self.mongodb_username}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}"
            else:
                # Không có authentication
                connection_string = f"mongodb://{self.mongodb_host}:{self.mongodb_port}/{self.mongodb_database}"
            
            # Kết nối với timeout
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Lấy database
            self.db = self.client[self.mongodb_database]
            
            logger.info(f"✅ Kết nối MongoDB thành công: {self.mongodb_host}:{self.mongodb_port}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Không thể kết nối MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối MongoDB: {e}")
            return False
    
    def is_connected(self) -> bool:
        """Kiểm tra kết nối MongoDB"""
        if not self.client:
            return False
        try:
            self.client.admin.command('ping')
            return True
        except:
            return False
    
    def get_collection(self, collection_name: str):
        """Lấy collection từ database"""
        if not self.is_connected():
            logger.error("MongoDB chưa được kết nối")
            return None
        
        if collection_name not in self.collections:
            logger.error(f"Collection '{collection_name}' không tồn tại")
            return None
        
        return self.db[self.collections[collection_name]]
    
    # ========================================
    # EMPLOYEE MANAGEMENT
    # ========================================
    
    def add_employee(self, employee_data: Dict[str, Any]) -> bool:
        """Thêm nhân viên mới"""
        try:
            collection = self.get_collection('employees')
            if not collection:
                return False
            
            # Thêm timestamp
            employee_data['created_at'] = datetime.now()
            employee_data['updated_at'] = datetime.now()
            
            result = collection.insert_one(employee_data)
            logger.info(f"✅ Đã thêm nhân viên: {employee_data.get('name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi thêm nhân viên: {e}")
            return False
    
    def get_employee(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin nhân viên theo ID"""
        try:
            collection = self.get_collection('employees')
            if not collection:
                return None
            
            employee = collection.find_one({'_id': employee_id})
            return employee
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy thông tin nhân viên: {e}")
            return None
    
    def get_all_employees(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả nhân viên"""
        try:
            collection = self.get_collection('employees')
            if not collection:
                return []
            
            employees = list(collection.find({}))
            return employees
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy danh sách nhân viên: {e}")
            return []
    
    def update_employee(self, employee_id: str, update_data: Dict[str, Any]) -> bool:
        """Cập nhật thông tin nhân viên"""
        try:
            collection = self.get_collection('employees')
            if not collection:
                return False
            
            # Thêm timestamp
            update_data['updated_at'] = datetime.now()
            
            result = collection.update_one(
                {'_id': employee_id},
                {'$set': update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Đã cập nhật nhân viên: {employee_id}")
                return True
            else:
                logger.warning(f"⚠️  Không tìm thấy nhân viên để cập nhật: {employee_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Lỗi cập nhật nhân viên: {e}")
            return False
    
    def delete_employee(self, employee_id: str) -> bool:
        """Xóa nhân viên"""
        try:
            collection = self.get_collection('employees')
            if not collection:
                return False
            
            result = collection.delete_one({'_id': employee_id})
            
            if result.deleted_count > 0:
                logger.info(f"✅ Đã xóa nhân viên: {employee_id}")
                return True
            else:
                logger.warning(f"⚠️  Không tìm thấy nhân viên để xóa: {employee_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Lỗi xóa nhân viên: {e}")
            return False
    
    # ========================================
    # FACE EMBEDDINGS MANAGEMENT
    # ========================================
    
    def save_face_embedding(self, employee_id: str, embedding: List[float], face_image_path: str = None) -> bool:
        """Lưu face embedding cho nhân viên"""
        try:
            collection = self.get_collection('face_embeddings')
            if not collection:
                return False
            
            embedding_data = {
                'employee_id': employee_id,
                'embedding': embedding,
                'face_image_path': face_image_path,
                'created_at': datetime.now(),
                'model': 'deepface'  # Có thể thay đổi theo model sử dụng
            }
            
            result = collection.insert_one(embedding_data)
            logger.info(f"✅ Đã lưu face embedding cho nhân viên: {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu face embedding: {e}")
            return False
    
    def get_face_embeddings(self, employee_id: str = None) -> List[Dict[str, Any]]:
        """Lấy face embeddings"""
        try:
            collection = self.get_collection('face_embeddings')
            if not collection:
                return []
            
            if employee_id:
                embeddings = list(collection.find({'employee_id': employee_id}))
            else:
                embeddings = list(collection.find({}))
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy face embeddings: {e}")
            return []
    
    def delete_face_embeddings(self, employee_id: str) -> bool:
        """Xóa face embeddings của nhân viên"""
        try:
            collection = self.get_collection('face_embeddings')
            if not collection:
                return False
            
            result = collection.delete_many({'employee_id': employee_id})
            logger.info(f"✅ Đã xóa {result.deleted_count} face embeddings cho nhân viên: {employee_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi xóa face embeddings: {e}")
            return False
    
    # ========================================
    # RECOGNITION LOGS
    # ========================================
    
    def log_recognition(self, recognition_data: Dict[str, Any]) -> bool:
        """Ghi log nhận diện khuôn mặt"""
        try:
            collection = self.get_collection('recognition_logs')
            if not collection:
                return False
            
            # Thêm timestamp
            recognition_data['timestamp'] = datetime.now()
            
            result = collection.insert_one(recognition_data)
            logger.info(f"✅ Đã ghi log nhận diện: {recognition_data.get('employee_id', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi ghi log nhận diện: {e}")
            return False
    
    def get_recognition_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Lấy logs nhận diện gần đây"""
        try:
            collection = self.get_collection('recognition_logs')
            if not collection:
                return []
            
            logs = list(collection.find({}).sort('timestamp', -1).limit(limit))
            return logs
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy recognition logs: {e}")
            return []
    
    # ========================================
    # TRAINING LOGS
    # ========================================
    
    def log_training(self, training_data: Dict[str, Any]) -> bool:
        """Ghi log training"""
        try:
            collection = self.get_collection('training_logs')
            if not collection:
                return False
            
            # Thêm timestamp
            training_data['timestamp'] = datetime.now()
            
            result = collection.insert_one(training_data)
            logger.info(f"✅ Đã ghi log training: {training_data.get('model_name', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi ghi log training: {e}")
            return False
    
    def get_training_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lấy logs training gần đây"""
        try:
            collection = self.get_collection('training_logs')
            if not collection:
                return []
            
            logs = list(collection.find({}).sort('timestamp', -1).limit(limit))
            return logs
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy training logs: {e}")
            return []
    
    # ========================================
    # DATABASE UTILITIES
    # ========================================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Lấy thống kê database"""
        try:
            if not self.is_connected():
                return {}
            
            stats = {
                'database_name': self.mongodb_database,
                'collections': {},
                'total_size': 0
            }
            
            for collection_name, collection_real_name in self.collections.items():
                collection = self.db[collection_real_name]
                count = collection.count_documents({})
                stats['collections'][collection_name] = count
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy database stats: {e}")
            return {}
    
    def clear_collection(self, collection_name: str) -> bool:
        """Xóa tất cả dữ liệu trong collection"""
        try:
            collection = self.get_collection(collection_name)
            if not collection:
                return False
            
            result = collection.delete_many({})
            logger.info(f"✅ Đã xóa {result.deleted_count} documents từ collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi xóa collection: {e}")
            return False
    
    def close_connection(self):
        """Đóng kết nối MongoDB"""
        if self.client:
            self.client.close()
            logger.info("🔌 Đã đóng kết nối MongoDB")

# Global instance
db_manager = MongoDBManager()

def get_db_manager() -> MongoDBManager:
    """Lấy instance MongoDB manager"""
    return db_manager 