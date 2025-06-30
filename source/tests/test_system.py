"""
Script test hệ thống nhận diện khuôn mặt
Kiểm tra các thành phần chính với YOLOv7 chính thức
"""

import sys
import os
from pathlib import Path
import logging

# Thêm đường dẫn để import các module
sys.path.append(str(Path(__file__).parent.parent))

from config.config import YOLOV7_CONFIG, MODEL_CONFIG, DATA_CONFIG
from utils.data_processor import DataProcessor
from utils.face_utils import FaceProcessor
from utils.image_utils import ImageProcessor

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_yolov7_availability():
    """Test YOLOv7 repository có sẵn không"""
    print("🔍 Kiểm tra YOLOv7 repository...")
    
    yolov7_path = YOLOV7_CONFIG['repo_path']
    
    if not yolov7_path.exists():
        print("❌ YOLOv7 repository không tồn tại")
        return False
    
    # Kiểm tra các file cần thiết
    required_files = YOLOV7_CONFIG['required_files']
    missing_files = []
    
    for file_path in required_files:
        full_path = yolov7_path / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Thiếu các file YOLOv7: {missing_files}")
        return False
    
    print("✅ YOLOv7 repository đã sẵn sàng")
    return True

def test_config():
    """Test cấu hình hệ thống"""
    print("🔍 Kiểm tra cấu hình hệ thống...")
    
    # Kiểm tra các thư mục cần thiết
    required_dirs = [
        DATA_CONFIG['raw_images_dir'],
        DATA_CONFIG['yolo_dataset_dir'],
        MODEL_CONFIG['face_embeddings_dir'],
        Path('models/pretrained'),
        Path('models/trained'),
        Path('logs'),
        Path('temp'),
        Path('outputs')
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        print(f"❌ Thiếu các thư mục: {missing_dirs}")
        return False
    
    print("✅ Cấu hình hệ thống OK")
    return True

def test_data_processor():
    """Test DataProcessor"""
    print("🔍 Kiểm tra DataProcessor...")
    
    try:
        data_processor = DataProcessor()
        
        # Test load metadata
        if DATA_CONFIG['metadata_file'].exists():
            metadata = data_processor.load_metadata()
            print(f"✅ Metadata loaded: {len(metadata)} records")
        else:
            print("⚠️  File metadata.csv không tồn tại (cần tạo)")
        
        # Test validate data
        integrity = data_processor.validate_data_integrity()
        print(f"✅ Data integrity check: {integrity['total_valid']} valid files")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi DataProcessor: {e}")
        return False

def test_face_processor():
    """Test FaceProcessor"""
    print("🔍 Kiểm tra FaceProcessor...")
    
    try:
        face_processor = FaceProcessor()
        print("✅ FaceProcessor khởi tạo thành công")
        
        # Test DeepFace import
        try:
            from deepface import DeepFace
            print("✅ DeepFace import OK")
        except ImportError:
            print("❌ DeepFace chưa được cài đặt")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi FaceProcessor: {e}")
        return False

def test_image_processor():
    """Test ImageProcessor"""
    print("🔍 Kiểm tra ImageProcessor...")
    
    try:
        image_processor = ImageProcessor()
        print("✅ ImageProcessor khởi tạo thành công")
        
        # Test OpenCV
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi ImageProcessor: {e}")
        return False

def test_pretrained_weights():
    """Test pretrained weights"""
    print("🔍 Kiểm tra pretrained weights...")
    
    weights_path = MODEL_CONFIG['yolov7_weights']
    
    if weights_path.exists():
        print(f"✅ YOLOv7 weights tồn tại: {weights_path}")
        return True
    else:
        print(f"⚠️  YOLOv7 weights không tồn tại: {weights_path}")
        print("   Chạy: wget -O models/pretrained/yolov7.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt")
        return False

def test_training_ready():
    """Test sẵn sàng cho training"""
    print("🔍 Kiểm tra sẵn sàng cho training...")
    
    # Kiểm tra dataset
    if DATA_CONFIG['raw_images_dir'].exists():
        image_files = list(DATA_CONFIG['raw_images_dir'].glob('*.jpg')) + \
                     list(DATA_CONFIG['raw_images_dir'].glob('*.png'))
        print(f"✅ Raw images: {len(image_files)} files")
    else:
        print("⚠️  Thư mục raw_images không tồn tại")
    
    # Kiểm tra metadata
    if DATA_CONFIG['metadata_file'].exists():
        print("✅ Metadata file tồn tại")
    else:
        print("⚠️  Metadata file không tồn tại")
    
    return True

def main():
    """Main test function"""
    print("🚀 Bắt đầu test hệ thống nhận diện khuôn mặt")
    print("=" * 50)
    
    tests = [
        ("YOLOv7 Availability", test_yolov7_availability),
        ("System Config", test_config),
        ("DataProcessor", test_data_processor),
        ("FaceProcessor", test_face_processor),
        ("ImageProcessor", test_image_processor),
        ("Pretrained Weights", test_pretrained_weights),
        ("Training Ready", test_training_ready),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Lỗi trong test {test_name}: {e}")
            results.append((test_name, False))
    
    # Tổng kết
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ TEST")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tổng kết: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Tất cả tests đều PASS! Hệ thống sẵn sàng sử dụng.")
    else:
        print("⚠️  Một số tests FAIL. Vui lòng kiểm tra và sửa lỗi.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 