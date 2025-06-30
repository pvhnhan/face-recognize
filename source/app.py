"""
Flask Application chính cho hệ thống nhận diện khuôn mặt
Chứa các endpoint API và cấu hình ứng dụng
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_restx import Api, Resource, fields, Namespace
import logging
import os
from pathlib import Path
import traceback

from config.config import FLASK_CONFIG, LOGGING_CONFIG
from utils.face_utils import FaceProcessor
from utils.image_utils import ImageProcessor
from api.face_recognition import face_recognition_bp, ns as face_ns

# Thiết lập logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    """Tạo Flask application"""
    app = Flask(__name__)
    
    # Cấu hình Flask
    app.config['MAX_CONTENT_LENGTH'] = FLASK_CONFIG['max_file_size']
    
    # CORS
    CORS(app)
    
    # Khởi tạo Flask-RESTX API với Swagger UI
    api = Api(
        app,
        version='1.0.0',
        title='Face Recognition API',
        description='API nhận diện khuôn mặt sử dụng YOLOv7 và DeepFace',
        doc='/docs',
        default='api',
        default_label='Face Recognition Endpoints'
    )
    
    # Thêm namespace từ face_recognition.py
    api.add_namespace(face_ns)
    
    # Tạo namespace cho các endpoint cơ bản
    ns = Namespace('system', description='System operations')
    api.add_namespace(ns)
    
    # Định nghĩa models cho Swagger
    health_model = ns.model('Health', {
        'status': fields.String(description='Trạng thái hệ thống'),
        'embeddings': fields.Raw(description='Thống kê embeddings'),
        'message': fields.String(description='Thông báo')
    })
    
    error_model = ns.model('Error', {
        'error': fields.String(description='Mô tả lỗi'),
        'status': fields.String(description='Trạng thái lỗi')
    })
    
    # Đăng ký blueprints
    app.register_blueprint(face_recognition_bp, url_prefix='/api')
    
    # Routes cơ bản
    @app.route('/')
    def index():
        """Trang chủ"""
        return jsonify({
            'message': 'Hệ thống nhận diện khuôn mặt',
            'version': '1.0.0',
            'endpoints': {
                'face_recognition': '/api/face-recognition',
                'health': '/health',
                'docs': '/docs'
            }
        })
    
    @ns.route('/health')
    class HealthCheck(Resource):
        @ns.doc('health_check')
        @ns.response(200, 'Success', health_model)
        @ns.response(500, 'Error', error_model)
        def get(self):
            """Kiểm tra sức khỏe hệ thống"""
            try:
                # Kiểm tra các thành phần cần thiết
                face_processor = FaceProcessor()
                image_processor = ImageProcessor()
                
                # Kiểm tra embeddings
                embeddings_stats = face_processor.get_embedding_statistics()
                
                return {
                    'status': 'healthy',
                    'embeddings': embeddings_stats,
                    'message': 'Hệ thống hoạt động bình thường'
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    'status': 'unhealthy',
                    'error': str(e)
                }, 500
    
    @ns.route('/statistics')
    class Statistics(Resource):
        @ns.doc('get_statistics')
        @ns.response(200, 'Success')
        def get(self):
            """Lấy thống kê hệ thống"""
            try:
                face_processor = FaceProcessor()
                stats = face_processor.get_embedding_statistics()
                
                return {
                    'total_embeddings': stats.get('total', 0),
                    'total_persons': stats.get('unique_persons', 0),
                    'last_updated': stats.get('last_updated', 'N/A'),
                    'status': 'success'
                }
            except Exception as e:
                logger.error(f"Statistics failed: {e}")
                return {
                    'error': str(e),
                    'status': 'error'
                }, 500
    
    # Error handlers
    @app.errorhandler(413)
    def too_large(e):
        """Xử lý lỗi file quá lớn"""
        return jsonify({
            'error': 'File quá lớn',
            'max_size': f"{FLASK_CONFIG['max_file_size'] / (1024*1024):.1f}MB",
            'status': 'error'
        }), 413
    
    @app.errorhandler(404)
    def not_found(e):
        """Xử lý lỗi 404"""
        return jsonify({
            'error': 'Endpoint không tồn tại',
            'status': 'error'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        """Xử lý lỗi 500"""
        logger.error(f"Internal server error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Lỗi nội bộ server',
            'status': 'error'
        }), 500
    
    return app

def main():
    """Hàm main để chạy Flask app"""
    app = create_app()
    
    # Lấy cấu hình từ environment
    host = os.getenv('FLASK_HOST', FLASK_CONFIG['host'])
    port = int(os.getenv('FLASK_PORT', FLASK_CONFIG['port']))
    debug = os.getenv('FLASK_DEBUG', FLASK_CONFIG['debug'])
    
    logger.info(f"Khởi động app tại {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    # Chạy app
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main() 