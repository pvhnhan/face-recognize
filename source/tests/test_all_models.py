"""
Script test tất cả các models của DeepFace để so sánh hiệu suất
"""

import sys
from pathlib import Path
import logging
import time
import numpy as np

# Thêm thư mục gốc vào Python path
sys.path.append(str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_deepface_models():
    """Test tất cả các models của DeepFace"""
    try:
        from deepface import DeepFace
        import numpy as np
        
        logger.info("✅ DeepFace import thành công")
        
        # Tạo ảnh test
        test_image = np.ones((160, 160, 3), dtype=np.uint8) * 128
        
        # Danh sách các models để test
        models = [
            'VGG-Face',
            'Facenet', 
            'Facenet512',
            'OpenFace',
            'DeepID',
            'ArcFace',
            'SFace'
        ]
        
        results = {}
        
        for model_name in models:
            try:
                logger.info(f"🧪 Testing model: {model_name}")
                start_time = time.time()
                
                embedding = DeepFace.represent(
                    img_path=test_image,
                    model_name=model_name,
                    enforce_detection=False
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                if isinstance(embedding, list) and len(embedding) > 0:
                    embedding_vector = embedding[0]['embedding']
                    embedding_dim = len(embedding_vector)
                    
                    results[model_name] = {
                        'status': '✅ Thành công',
                        'dimension': embedding_dim,
                        'time': processing_time,
                        'embedding_sample': embedding_vector[:5]  # 5 giá trị đầu tiên
                    }
                    
                    logger.info(f"✅ {model_name}: {embedding_dim} chiều, {processing_time:.3f}s")
                else:
                    results[model_name] = {
                        'status': '❌ Lỗi',
                        'error': 'Không nhận được embedding hợp lệ'
                    }
                    logger.error(f"❌ {model_name}: Không nhận được embedding hợp lệ")
                    
            except Exception as e:
                results[model_name] = {
                    'status': '❌ Lỗi',
                    'error': str(e)
                }
                logger.error(f"❌ {model_name}: {e}")
        
        # In kết quả tổng hợp
        logger.info("\n" + "="*60)
        logger.info("📊 KẾT QUẢ TEST TẤT CẢ MODELS")
        logger.info("="*60)
        
        for model_name, result in results.items():
            if result['status'] == '✅ Thành công':
                logger.info(f"{model_name:12} | {result['dimension']:3} chiều | {result['time']:6.3f}s | ✅")
            else:
                logger.error(f"{model_name:12} | {result['error']}")
        
        # So sánh hiệu suất
        successful_models = {k: v for k, v in results.items() if v['status'] == '✅ Thành công'}
        
        if successful_models:
            logger.info("\n" + "="*60)
            logger.info("🏆 SO SÁNH HIỆU SUẤT")
            logger.info("="*60)
            
            # Sắp xếp theo thời gian xử lý
            sorted_by_time = sorted(successful_models.items(), key=lambda x: x[1]['time'])
            
            for i, (model_name, result) in enumerate(sorted_by_time, 1):
                logger.info(f"{i}. {model_name:12} | {result['time']:6.3f}s | {result['dimension']:3} chiều")
            
            # Khuyến nghị model tốt nhất
            best_model = sorted_by_time[0][0]
            logger.info(f"\n🎯 KHUYẾN NGHỊ: Sử dụng {best_model} (nhanh nhất)")
            
            # Nếu Facenet512 có sẵn và không quá chậm, khuyến nghị sử dụng
            if 'Facenet512' in successful_models:
                facenet512_time = successful_models['Facenet512']['time']
                if facenet512_time < 1.0:  # Dưới 1 giây
                    logger.info(f"💡 Facenet512 ({facenet512_time:.3f}s) là lựa chọn tốt cho độ chính xác cao")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi khi test DeepFace models: {e}")
        return False

def test_model_compatibility():
    """Test tính tương thích của các models"""
    logger.info("\n🔧 Kiểm tra tính tương thích...")
    
    try:
        from deepface import DeepFace
        import numpy as np
        
        # Tạo 2 ảnh test khác nhau
        img1 = np.ones((160, 160, 3), dtype=np.uint8) * 100
        img2 = np.ones((160, 160, 3), dtype=np.uint8) * 150
        
        # Test với Facenet512 (model được khuyến nghị)
        logger.info("🧪 Test tính tương thích với Facenet512...")
        
        embedding1 = DeepFace.represent(
            img_path=img1,
            model_name='Facenet512',
            enforce_detection=False
        )
        
        embedding2 = DeepFace.represent(
            img_path=img2,
            model_name='Facenet512',
            enforce_detection=False
        )
        
        if isinstance(embedding1, list) and isinstance(embedding2, list):
            vec1 = np.array(embedding1[0]['embedding'])
            vec2 = np.array(embedding2[0]['embedding'])
            
            # Tính cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
            
            logger.info(f"✅ Facenet512 hoạt động bình thường")
            logger.info(f"📊 Similarity giữa 2 ảnh test: {similarity:.4f}")
            
            return True
        else:
            logger.error("❌ Không thể trích xuất embeddings")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi khi test tính tương thích: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Bắt đầu test tất cả models của DeepFace...")
    
    # Test tất cả models
    success1 = test_deepface_models()
    
    # Test tính tương thích
    success2 = test_model_compatibility()
    
    if success1 and success2:
        logger.info("\n🎉 Tất cả tests đều thành công!")
        logger.info("📝 Hệ thống sẵn sàng sử dụng Facenet512 model")
    else:
        logger.error("\n❌ Có lỗi trong quá trình test")
        sys.exit(1) 