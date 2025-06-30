"""
Script test tích hợp hệ thống nhận diện khuôn mặt
"""

import os
import sys
import time
import json
import requests
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_processor import DataProcessor
from utils.face_utils import FaceProcessor
from utils.image_utils import ImageProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.data_processor = DataProcessor()
        self.face_processor = FaceProcessor()
        self.image_processor = ImageProcessor()
        
        self.result_dir = Path(__file__).parent / "result"
        self.result_dir.mkdir(exist_ok=True)
        
        self.test_data_dir = self.result_dir / "integration_test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        self.integration_results = {
            'timestamp': datetime.now().isoformat(),
            'base_url': base_url,
            'tests': [],
            'summary': {}
        }
        
        logger.info(f"Khởi tạo IntegrationTester với base_url: {base_url}")
    
    def setup_test_data(self):
        """Thiết lập dữ liệu test"""
        logger.info("Thiết lập dữ liệu test...")
        
        employees = [
            {'employee_id': 'EMP001', 'full_name': 'Nguyễn Văn A'},
            {'employee_id': 'EMP002', 'full_name': 'Trần Thị B'},
            {'employee_id': 'EMP003', 'full_name': 'Lê Văn C'}
        ]
        
        test_images = []
        
        for emp in employees:
            emp_dir = self.test_data_dir / emp['employee_id']
            emp_dir.mkdir(exist_ok=True)
            
            for i in range(3):
                image_path = emp_dir / f"image_{i+1}.jpg"
                image = np.ones((400, 300, 3), dtype=np.uint8) * 255
                
                center = (150, 200)
                radius = 50
                cv2.circle(image, center, radius, (200, 200, 200), -1)
                
                eye_radius = radius // 4
                left_eye = (center[0] - radius // 3, center[1] - radius // 4)
                right_eye = (center[0] + radius // 3, center[1] - radius // 4)
                cv2.circle(image, left_eye, eye_radius, (0, 0, 0), -1)
                cv2.circle(image, right_eye, eye_radius, (0, 0, 0), -1)
                
                mouth_start = (center[0] - radius // 3, center[1] + radius // 3)
                mouth_end = (center[0] + radius // 3, center[1] + radius // 3)
                cv2.line(image, mouth_start, mouth_end, (0, 0, 0), 2)
                
                cv2.putText(image, emp['full_name'], (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                cv2.imwrite(str(image_path), image)
                test_images.append(str(image_path))
        
        metadata_file = self.test_data_dir / "test_metadata.csv"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("employee_id,full_name,image_path\n")
            for emp in employees:
                for i in range(3):
                    image_path = f"{emp['employee_id']}/image_{i+1}.jpg"
                    f.write(f"{emp['employee_id']},{emp['full_name']},{image_path}\n")
        
        return {
            'employees': employees,
            'test_images': test_images,
            'metadata_file': str(metadata_file)
        }
    
    def test_data_processing(self, test_data_info):
        """Test xử lý dữ liệu"""
        logger.info("Testing data processing...")
        
        test_result = {
            'test_name': 'data_processing',
            'start_time': time.time(),
            'status': 'failed',
            'details': {}
        }
        
        try:
            metadata = self.data_processor.load_metadata(test_data_info['metadata_file'])
            valid_metadata = self.data_processor.validate_metadata(metadata)
            train_data, val_data, test_data = self.data_processor.split_dataset(valid_metadata)
            
            test_result['details'] = {
                'metadata_count': len(metadata),
                'valid_count': len(valid_metadata),
                'train_count': len(train_data),
                'val_count': len(val_data),
                'test_count': len(test_data)
            }
            
            test_result['status'] = 'passed'
            logger.info("Data processing test: PASSED")
        
        except Exception as e:
            test_result['details']['error'] = str(e)
            logger.error(f"Data processing test: FAILED - {e}")
        
        test_result['end_time'] = time.time()
        test_result['duration'] = test_result['end_time'] - test_result['start_time']
        return test_result
    
    def test_face_detection(self, test_data_info):
        """Test phát hiện khuôn mặt"""
        logger.info("Testing face detection...")
        
        test_result = {
            'test_name': 'face_detection',
            'start_time': time.time(),
            'status': 'failed',
            'details': {}
        }
        
        try:
            test_image = test_data_info['test_images'][0]
            faces = self.face_processor.detect_faces(test_image)
            
            test_result['details'] = {
                'faces_detected': len(faces),
                'detection_works': len(faces) > 0
            }
            
            if len(faces) > 0:
                embedding = self.face_processor.extract_face_embedding(test_image, faces[0])
                test_result['details']['embedding_extracted'] = embedding is not None
                test_result['details']['embedding_dimension'] = len(embedding) if embedding is not None else 0
            
            test_result['status'] = 'passed'
            logger.info("Face detection test: PASSED")
        
        except Exception as e:
            test_result['details']['error'] = str(e)
            logger.error(f"Face detection test: FAILED - {e}")
        
        test_result['end_time'] = time.time()
        test_result['duration'] = test_result['end_time'] - test_result['start_time']
        return test_result
    
    def test_api_integration(self, test_data_info):
        """Test tích hợp API"""
        logger.info("Testing API integration...")
        
        test_result = {
            'test_name': 'api_integration',
            'start_time': time.time(),
            'status': 'failed',
            'details': {}
        }
        
        try:
            # Health check
            response = requests.get(f"{self.base_url}/health", timeout=10)
            test_result['details']['health_check'] = response.status_code == 200
            
            if response.status_code != 200:
                raise Exception(f"Health check failed: {response.status_code}")
            
            # Face recognition test
            successful_recognition = 0
            total_tests = min(3, len(test_data_info['test_images']))
            
            for i in range(total_tests):
                image_path = test_data_info['test_images'][i]
                
                with open(image_path, 'rb') as f:
                    files = {'image': (Path(image_path).name, f, 'image/jpeg')}
                    response = requests.post(
                        f"{self.base_url}/api/face-recognition",
                        files=files,
                        timeout=30
                    )
                
                if response.status_code == 200:
                    successful_recognition += 1
                    data = response.json()
                    logger.info(f"API test {i+1}: {data.get('full_name', 'Unknown')}")
            
            test_result['details'] = {
                'api_tests_total': total_tests,
                'api_tests_successful': successful_recognition,
                'api_success_rate': (successful_recognition / total_tests * 100) if total_tests > 0 else 0
            }
            
            test_result['status'] = 'passed'
            logger.info("API integration test: PASSED")
        
        except Exception as e:
            test_result['details']['error'] = str(e)
            logger.error(f"API integration test: FAILED - {e}")
        
        test_result['end_time'] = time.time()
        test_result['duration'] = test_result['end_time'] - test_result['start_time']
        return test_result
    
    def run_integration_tests(self):
        """Chạy tất cả integration tests"""
        logger.info("Bắt đầu chạy integration tests...")
        
        test_data_info = self.setup_test_data()
        
        # Test data processing
        self.integration_results['tests'].append(
            self.test_data_processing(test_data_info)
        )
        
        # Test face detection
        self.integration_results['tests'].append(
            self.test_face_detection(test_data_info)
        )
        
        # Test API integration
        self.integration_results['tests'].append(
            self.test_api_integration(test_data_info)
        )
        
        # Calculate summary
        total_tests = len(self.integration_results['tests'])
        passed_tests = len([t for t in self.integration_results['tests'] if t['status'] == 'passed'])
        
        self.integration_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': sum(t.get('duration', 0) for t in self.integration_results['tests'])
        }
        
        # Save results
        result_file = self.result_dir / f"integration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.integration_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Kết quả integration test đã được lưu tại: {result_file}")
        return self.integration_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test tích hợp hệ thống nhận diện khuôn mặt')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                       help='URL của Flask API')
    
    args = parser.parse_args()
    
    tester = IntegrationTester(args.url)
    
    try:
        results = tester.run_integration_tests()
        
        print("\n" + "="*50)
        print("INTEGRATION TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Total Duration: {results['summary']['total_duration']:.2f}s")
        print("="*50)
        
    except Exception as e:
        print(f"\nLỗi khi chạy integration tests: {e}")

if __name__ == '__main__':
    main() 