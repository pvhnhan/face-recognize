"""
Script test hiệu năng hệ thống nhận diện khuôn mặt
Đo lường thời gian phản hồi, throughput và tài nguyên sử dụng
"""

import os
import sys
import time
import json
import requests
import threading
import psutil
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Thêm đường dẫn để import các module
sys.path.append(str(Path(__file__).parent.parent))

from utils.image_utils import ImageProcessor

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    """Lớp test hiệu năng hệ thống nhận diện khuôn mặt"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Khởi tạo PerformanceTester
        
        Args:
            base_url: URL của Flask API
        """
        self.base_url = base_url
        self.image_processor = ImageProcessor()
        
        # Tạo thư mục kết quả
        self.result_dir = Path(__file__).parent / "result"
        self.result_dir.mkdir(exist_ok=True)
        
        # Tạo thư mục test images
        self.test_images_dir = self.result_dir / "performance_images"
        self.test_images_dir.mkdir(exist_ok=True)
        
        # Kết quả test
        self.performance_results = {
            'timestamp': datetime.now().isoformat(),
            'base_url': base_url,
            'system_info': self._get_system_info(),
            'tests': [],
            'summary': {}
        }
        
        logger.info(f"Khởi tạo PerformanceTester với base_url: {base_url}")
        logger.info(f"Kết quả sẽ được lưu tại: {self.result_dir}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Lấy thông tin hệ thống"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'platform': sys.platform,
            'python_version': sys.version
        }
    
    def create_performance_test_images(self, count: int = 50) -> List[str]:
        """
        Tạo ảnh test cho performance testing
        
        Args:
            count: Số lượng ảnh cần tạo
            
        Returns:
            List[str]: Danh sách đường dẫn ảnh test
        """
        logger.info(f"Tạo {count} ảnh test cho performance testing...")
        
        test_images = []
        
        for i in range(count):
            # Tạo ảnh với kích thước khác nhau
            width = np.random.randint(200, 800)
            height = np.random.randint(150, 600)
            
            # Tạo ảnh nền
            image = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # Thêm khuôn mặt giả
            face_count = np.random.randint(1, 4)
            for j in range(face_count):
                # Vị trí ngẫu nhiên
                x = np.random.randint(50, width - 50)
                y = np.random.randint(50, height - 50)
                
                # Vẽ khuôn mặt
                radius = np.random.randint(20, 50)
                cv2.circle(image, (x, y), radius, (200, 200, 200), -1)
                
                # Vẽ mắt
                eye_radius = radius // 4
                left_eye = (x - radius // 3, y - radius // 4)
                right_eye = (x + radius // 3, y - radius // 4)
                cv2.circle(image, left_eye, eye_radius, (0, 0, 0), -1)
                cv2.circle(image, right_eye, eye_radius, (0, 0, 0), -1)
                
                # Vẽ miệng
                mouth_start = (x - radius // 3, y + radius // 3)
                mouth_end = (x + radius // 3, y + radius // 3)
                cv2.line(image, mouth_start, mouth_end, (0, 0, 0), 2)
            
            # Lưu ảnh
            image_path = self.test_images_dir / f"perf_test_{i:03d}.jpg"
            cv2.imwrite(str(image_path), image)
            test_images.append(str(image_path))
        
        logger.info(f"Đã tạo {len(test_images)} ảnh test")
        return test_images
    
    def test_single_request_performance(self, image_path: str) -> Dict[str, Any]:
        """
        Test hiệu năng một request đơn lẻ
        
        Args:
            image_path: Đường dẫn ảnh test
            
        Returns:
            Dict[str, Any]: Kết quả test
        """
        result = {
            'image_file': Path(image_path).name,
            'file_size': os.path.getsize(image_path),
            'response_time': 0,
            'status_code': 0,
            'success': False,
            'error': None
        }
        
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (Path(image_path).name, f, 'image/jpeg')}
                
                start_time = time.time()
                response = requests.post(
                    f'{self.base_url}/api/face-recognition',
                    files=files,
                    timeout=60
                )
                end_time = time.time()
                
                result['response_time'] = end_time - start_time
                result['status_code'] = response.status_code
                result['success'] = response.status_code == 200
                
                if response.status_code == 200:
                    data = response.json()
                    result['face_count'] = data.get('face_count', 0)
                    result['similarity_score'] = data.get('similarity_score', 0)
                else:
                    result['error'] = f'HTTP {response.status_code}'
        
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def test_sequential_performance(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Test hiệu năng sequential requests
        
        Args:
            image_paths: Danh sách đường dẫn ảnh test
            
        Returns:
            Dict[str, Any]: Kết quả test
        """
        logger.info(f"Testing sequential performance với {len(image_paths)} ảnh...")
        
        test_result = {
            'test_name': 'sequential_performance',
            'total_requests': len(image_paths),
            'start_time': time.time(),
            'results': [],
            'summary': {}
        }
        
        # Thực hiện requests tuần tự
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing image {i+1}/{len(image_paths)}: {Path(image_path).name}")
            result = self.test_single_request_performance(image_path)
            test_result['results'].append(result)
        
        test_result['end_time'] = time.time()
        test_result['total_time'] = test_result['end_time'] - test_result['start_time']
        
        # Tính toán summary
        successful_requests = [r for r in test_result['results'] if r['success']]
        response_times = [r['response_time'] for r in successful_requests]
        
        test_result['summary'] = {
            'total_requests': len(image_paths),
            'successful_requests': len(successful_requests),
            'failed_requests': len(image_paths) - len(successful_requests),
            'success_rate': len(successful_requests) / len(image_paths) * 100,
            'total_time': test_result['total_time'],
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'throughput': len(successful_requests) / test_result['total_time'] if test_result['total_time'] > 0 else 0
        }
        
        logger.info(f"Sequential test completed: {test_result['summary']['successful_requests']}/{len(image_paths)} successful")
        return test_result
    
    def test_concurrent_performance(self, image_paths: List[str], max_workers: int = 5) -> Dict[str, Any]:
        """
        Test hiệu năng concurrent requests
        
        Args:
            image_paths: Danh sách đường dẫn ảnh test
            max_workers: Số lượng worker threads tối đa
            
        Returns:
            Dict[str, Any]: Kết quả test
        """
        logger.info(f"Testing concurrent performance với {len(image_paths)} ảnh, {max_workers} workers...")
        
        test_result = {
            'test_name': 'concurrent_performance',
            'total_requests': len(image_paths),
            'max_workers': max_workers,
            'start_time': time.time(),
            'results': [],
            'summary': {}
        }
        
        # Thực hiện requests đồng thời
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {
                executor.submit(self.test_single_request_performance, image_path): image_path
                for image_path in image_paths
            }
            
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result = future.result()
                    test_result['results'].append(result)
                    logger.info(f"Completed: {Path(image_path).name} - {result['response_time']:.3f}s")
                except Exception as e:
                    logger.error(f"Error processing {Path(image_path).name}: {e}")
                    test_result['results'].append({
                        'image_file': Path(image_path).name,
                        'success': False,
                        'error': str(e)
                    })
        
        test_result['end_time'] = time.time()
        test_result['total_time'] = test_result['end_time'] - test_result['start_time']
        
        # Tính toán summary
        successful_requests = [r for r in test_result['results'] if r['success']]
        response_times = [r['response_time'] for r in successful_requests]
        
        test_result['summary'] = {
            'total_requests': len(image_paths),
            'successful_requests': len(successful_requests),
            'failed_requests': len(image_paths) - len(successful_requests),
            'success_rate': len(successful_requests) / len(image_paths) * 100,
            'total_time': test_result['total_time'],
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'throughput': len(successful_requests) / test_result['total_time'] if test_result['total_time'] > 0 else 0
        }
        
        logger.info(f"Concurrent test completed: {test_result['summary']['successful_requests']}/{len(image_paths)} successful")
        return test_result
    
    def test_load_performance(self, image_paths: List[str], duration: int = 60) -> Dict[str, Any]:
        """
        Test hiệu năng dưới tải liên tục
        
        Args:
            image_paths: Danh sách đường dẫn ảnh test
            duration: Thời gian test (giây)
            
        Returns:
            Dict[str, Any]: Kết quả test
        """
        logger.info(f"Testing load performance trong {duration} giây...")
        
        test_result = {
            'test_name': 'load_performance',
            'duration': duration,
            'start_time': time.time(),
            'results': [],
            'summary': {}
        }
        
        end_time = time.time() + duration
        request_count = 0
        
        while time.time() < end_time:
            # Chọn ảnh ngẫu nhiên
            image_path = np.random.choice(image_paths)
            result = self.test_single_request_performance(image_path)
            result['request_id'] = request_count
            test_result['results'].append(result)
            request_count += 1
            
            # Delay ngẫu nhiên để mô phỏng tải thực tế
            time.sleep(np.random.uniform(0.1, 0.5))
        
        test_result['end_time'] = time.time()
        test_result['total_time'] = test_result['end_time'] - test_result['start_time']
        test_result['total_requests'] = request_count
        
        # Tính toán summary
        successful_requests = [r for r in test_result['results'] if r['success']]
        response_times = [r['response_time'] for r in successful_requests]
        
        test_result['summary'] = {
            'total_requests': request_count,
            'successful_requests': len(successful_requests),
            'failed_requests': request_count - len(successful_requests),
            'success_rate': len(successful_requests) / request_count * 100,
            'total_time': test_result['total_time'],
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'median_response_time': statistics.median(response_times) if response_times else 0,
            'throughput': len(successful_requests) / test_result['total_time'] if test_result['total_time'] > 0 else 0,
            'requests_per_second': request_count / test_result['total_time'] if test_result['total_time'] > 0 else 0
        }
        
        logger.info(f"Load test completed: {test_result['summary']['successful_requests']}/{request_count} successful")
        return test_result
    
    def monitor_system_resources(self, duration: int = 60) -> Dict[str, Any]:
        """
        Monitor tài nguyên hệ thống
        
        Args:
            duration: Thời gian monitor (giây)
            
        Returns:
            Dict[str, Any]: Kết quả monitor
        """
        logger.info(f"Monitoring system resources trong {duration} giây...")
        
        monitor_result = {
            'test_name': 'system_resources',
            'duration': duration,
            'start_time': time.time(),
            'samples': [],
            'summary': {}
        }
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            sample = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used': psutil.virtual_memory().used,
                'memory_available': psutil.virtual_memory().available
            }
            monitor_result['samples'].append(sample)
            time.sleep(1)
        
        monitor_result['end_time'] = time.time()
        monitor_result['total_time'] = monitor_result['end_time'] - monitor_result['start_time']
        
        # Tính toán summary
        cpu_percentages = [s['cpu_percent'] for s in monitor_result['samples']]
        memory_percentages = [s['memory_percent'] for s in monitor_result['samples']]
        
        monitor_result['summary'] = {
            'total_samples': len(monitor_result['samples']),
            'avg_cpu_percent': statistics.mean(cpu_percentages),
            'max_cpu_percent': max(cpu_percentages),
            'min_cpu_percent': min(cpu_percentages),
            'avg_memory_percent': statistics.mean(memory_percentages),
            'max_memory_percent': max(memory_percentages),
            'min_memory_percent': min(memory_percentages)
        }
        
        logger.info(f"System monitoring completed: {len(monitor_result['samples'])} samples")
        return monitor_result
    
    def run_performance_tests(self, image_count: int = 50, load_duration: int = 60) -> Dict[str, Any]:
        """
        Chạy tất cả performance tests
        
        Args:
            image_count: Số lượng ảnh test
            load_duration: Thời gian load test (giây)
            
        Returns:
            Dict[str, Any]: Kết quả tất cả tests
        """
        logger.info("Bắt đầu chạy performance tests...")
        
        # Tạo ảnh test
        test_images = self.create_performance_test_images(image_count)
        
        # Test 1: Sequential performance
        self.performance_results['tests'].append(
            self.test_sequential_performance(test_images)
        )
        
        # Test 2: Concurrent performance
        self.performance_results['tests'].append(
            self.test_concurrent_performance(test_images, max_workers=5)
        )
        
        # Test 3: Load performance
        self.performance_results['tests'].append(
            self.test_load_performance(test_images, duration=load_duration)
        )
        
        # Test 4: System resources monitoring
        self.performance_results['tests'].append(
            self.monitor_system_resources(duration=load_duration)
        )
        
        # Tính toán summary tổng thể
        self._calculate_performance_summary()
        
        # Lưu kết quả
        self._save_performance_results()
        
        logger.info("Hoàn thành performance tests!")
        return self.performance_results
    
    def _calculate_performance_summary(self):
        """Tính toán summary tổng thể của performance tests"""
        summary = {
            'total_tests': len(self.performance_results['tests']),
            'overall_throughput': 0,
            'avg_response_time': 0,
            'best_performance': None,
            'worst_performance': None
        }
        
        # Tính toán metrics tổng thể
        all_response_times = []
        all_throughputs = []
        
        for test in self.performance_results['tests']:
            if 'summary' in test and 'throughput' in test['summary']:
                all_throughputs.append(test['summary']['throughput'])
            if 'summary' in test and 'avg_response_time' in test['summary']:
                all_response_times.append(test['summary']['avg_response_time'])
        
        if all_throughputs:
            summary['overall_throughput'] = statistics.mean(all_throughputs)
            summary['best_performance'] = max(all_throughputs)
            summary['worst_performance'] = min(all_throughputs)
        
        if all_response_times:
            summary['avg_response_time'] = statistics.mean(all_response_times)
        
        self.performance_results['summary'] = summary
    
    def _save_performance_results(self):
        """Lưu kết quả performance tests"""
        # Lưu kết quả JSON
        result_file = self.result_dir / f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.performance_results, f, indent=2, ensure_ascii=False)
        
        # Tạo báo cáo HTML
        self._generate_performance_html_report()
        
        # Tạo báo cáo text
        self._generate_performance_text_report()
        
        logger.info(f"Kết quả performance test đã được lưu tại: {result_file}")
    
    def _generate_performance_html_report(self):
        """Tạo báo cáo HTML cho performance tests"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition System Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .test-result {{ margin: 20px 0; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }}
        .metric {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #007bff; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Face Recognition System Performance Report</h1>
        <p><strong>Timestamp:</strong> {self.performance_results['timestamp']}</p>
        <p><strong>Base URL:</strong> {self.performance_results['base_url']}</p>
    </div>
    
    <div class="summary">
        <h2>Performance Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{self.performance_results['summary']['overall_throughput']:.2f}</div>
                <div>Overall Throughput (req/s)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.performance_results['summary']['avg_response_time']:.3f}s</div>
                <div>Avg Response Time</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.performance_results['summary']['best_performance']:.2f}</div>
                <div>Best Throughput (req/s)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.performance_results['summary']['worst_performance']:.2f}</div>
                <div>Worst Throughput (req/s)</div>
            </div>
        </div>
    </div>
    
    <h2>Test Results</h2>
"""
        
        for test in self.performance_results['tests']:
            html_content += f"""
    <div class="test-result">
        <h3>{test['test_name']}</h3>
        <div class="metrics">
"""
            
            if 'summary' in test:
                summary = test['summary']
                if 'throughput' in summary:
                    html_content += f"""
            <div class="metric">
                <div class="metric-value">{summary['throughput']:.2f}</div>
                <div>Throughput (req/s)</div>
            </div>
"""
                if 'avg_response_time' in summary:
                    html_content += f"""
            <div class="metric">
                <div class="metric-value">{summary['avg_response_time']:.3f}s</div>
                <div>Avg Response Time</div>
            </div>
"""
                if 'success_rate' in summary:
                    html_content += f"""
            <div class="metric">
                <div class="metric-value">{summary['success_rate']:.1f}%</div>
                <div>Success Rate</div>
            </div>
"""
                if 'total_requests' in summary:
                    html_content += f"""
            <div class="metric">
                <div class="metric-value">{summary['total_requests']}</div>
                <div>Total Requests</div>
            </div>
"""
            
            html_content += """
        </div>
        <details>
            <summary>Detailed Results</summary>
            <pre>{}</pre>
        </details>
    </div>
""".format(json.dumps(test, indent=2, ensure_ascii=False))
        
        html_content += """
</body>
</html>
"""
        
        html_file = self.result_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Báo cáo performance HTML đã được tạo: {html_file}")
    
    def _generate_performance_text_report(self):
        """Tạo báo cáo text cho performance tests"""
        text_content = f"""
FACE RECOGNITION SYSTEM PERFORMANCE REPORT
==========================================

Timestamp: {self.performance_results['timestamp']}
Base URL: {self.performance_results['base_url']}

PERFORMANCE SUMMARY
------------------
Overall Throughput: {self.performance_results['summary']['overall_throughput']:.2f} req/s
Average Response Time: {self.performance_results['summary']['avg_response_time']:.3f}s
Best Performance: {self.performance_results['summary']['best_performance']:.2f} req/s
Worst Performance: {self.performance_results['summary']['worst_performance']:.2f} req/s

SYSTEM INFORMATION
-----------------
CPU Count: {self.performance_results['system_info']['cpu_count']}
Memory Total: {self.performance_results['system_info']['memory_total'] / (1024**3):.2f} GB
Memory Available: {self.performance_results['system_info']['memory_available'] / (1024**3):.2f} GB
Platform: {self.performance_results['system_info']['platform']}

DETAILED TEST RESULTS
--------------------
"""
        
        for test in self.performance_results['tests']:
            text_content += f"""
Test: {test['test_name']}
"""
            if 'summary' in test:
                summary = test['summary']
                for key, value in summary.items():
                    if isinstance(value, float):
                        text_content += f"{key}: {value:.3f}\n"
                    else:
                        text_content += f"{key}: {value}\n"
            text_content += "\n"
        
        text_file = self.result_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"Báo cáo performance text đã được tạo: {text_file}")

def main():
    """Hàm main để chạy performance tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test hiệu năng hệ thống nhận diện khuôn mặt')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                       help='URL của Flask API (default: http://localhost:5000)')
    parser.add_argument('--images', type=int, default=50,
                       help='Số lượng ảnh test (default: 50)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Thời gian load test (giây, default: 60)')
    
    args = parser.parse_args()
    
    # Khởi tạo tester
    tester = PerformanceTester(args.url)
    
    try:
        # Chạy performance tests
        results = tester.run_performance_tests(
            image_count=args.images,
            load_duration=args.duration
        )
        
        # In summary
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUMMARY")
        print("="*60)
        print(f"Overall Throughput: {results['summary']['overall_throughput']:.2f} req/s")
        print(f"Average Response Time: {results['summary']['avg_response_time']:.3f}s")
        print(f"Best Performance: {results['summary']['best_performance']:.2f} req/s")
        print(f"Worst Performance: {results['summary']['worst_performance']:.2f} req/s")
        print("="*60)
        
        print(f"\nKết quả chi tiết đã được lưu tại: {tester.result_dir}")
        
    except KeyboardInterrupt:
        print("\nPerformance test bị gián đoạn bởi người dùng")
    except Exception as e:
        print(f"\nLỗi khi chạy performance tests: {e}")
        logger.error(f"Lỗi khi chạy performance tests: {e}")

if __name__ == '__main__':
    main() 