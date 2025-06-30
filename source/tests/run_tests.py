#!/usr/bin/env python3
"""
Script chính để chạy tất cả tests hệ thống nhận diện khuôn mặt
Bao gồm: unit tests, system tests, performance tests
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRunner:
    """Lớp chạy tất cả tests"""
    
    def __init__(self):
        """Khởi tạo TestRunner"""
        self.test_dir = Path(__file__).parent
        self.result_dir = self.test_dir / "result"
        self.result_dir.mkdir(exist_ok=True)
        
        # Kết quả tổng thể
        self.overall_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': [],
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'success_rate': 0.0
            }
        }
        
        logger.info(f"Khởi tạo TestRunner")
        logger.info(f"Thư mục kết quả: {self.result_dir}")
    
    def run_unit_tests(self) -> dict:
        """Chạy unit tests"""
        logger.info("Bắt đầu chạy unit tests...")
        
        test_result = {
            'test_type': 'unit_tests',
            'start_time': time.time(),
            'status': 'failed',
            'details': {}
        }
        
        try:
            # Chạy pytest
            cmd = [sys.executable, '-m', 'pytest', 'test_api.py', '-v', '--tb=short']
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']
            test_result['return_code'] = result.returncode
            test_result['stdout'] = result.stdout
            test_result['stderr'] = result.stderr
            
            if result.returncode == 0:
                test_result['status'] = 'passed'
                logger.info("Unit tests: PASSED")
            else:
                logger.error("Unit tests: FAILED")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            test_result['status'] = 'timeout'
            test_result['details'] = {'error': 'Unit tests timeout after 5 minutes'}
            logger.error("Unit tests: TIMEOUT")
        except Exception as e:
            test_result['status'] = 'error'
            test_result['details'] = {'error': str(e)}
            logger.error(f"Unit tests: ERROR - {e}")
        
        return test_result
    
    def run_system_tests(self, api_url: str = "http://localhost:5000") -> dict:
        """Chạy system tests"""
        logger.info("Bắt đầu chạy system tests...")
        
        test_result = {
            'test_type': 'system_tests',
            'start_time': time.time(),
            'status': 'failed',
            'details': {}
        }
        
        try:
            # Chạy system test script
            cmd = [sys.executable, 'test_system.py', '--url', api_url]
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']
            test_result['return_code'] = result.returncode
            test_result['stdout'] = result.stdout
            test_result['stderr'] = result.stderr
            
            if result.returncode == 0:
                test_result['status'] = 'passed'
                logger.info("System tests: PASSED")
            else:
                logger.error("System tests: FAILED")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            test_result['status'] = 'timeout'
            test_result['details'] = {'error': 'System tests timeout after 10 minutes'}
            logger.error("System tests: TIMEOUT")
        except Exception as e:
            test_result['status'] = 'error'
            test_result['details'] = {'error': str(e)}
            logger.error(f"System tests: ERROR - {e}")
        
        return test_result
    
    def run_performance_tests(self, api_url: str = "http://localhost:5000", 
                            image_count: int = 50, duration: int = 60) -> dict:
        """Chạy performance tests"""
        logger.info("Bắt đầu chạy performance tests...")
        
        test_result = {
            'test_type': 'performance_tests',
            'start_time': time.time(),
            'status': 'failed',
            'details': {}
        }
        
        try:
            # Chạy performance test script
            cmd = [
                sys.executable, 'test_performance.py',
                '--url', api_url,
                '--images', str(image_count),
                '--duration', str(duration)
            ]
            result = subprocess.run(
                cmd,
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 phút
            )
            
            test_result['end_time'] = time.time()
            test_result['duration'] = test_result['end_time'] - test_result['start_time']
            test_result['return_code'] = result.returncode
            test_result['stdout'] = result.stdout
            test_result['stderr'] = result.stderr
            
            if result.returncode == 0:
                test_result['status'] = 'passed'
                logger.info("Performance tests: PASSED")
            else:
                logger.error("Performance tests: FAILED")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            test_result['status'] = 'timeout'
            test_result['details'] = {'error': 'Performance tests timeout after 30 minutes'}
            logger.error("Performance tests: TIMEOUT")
        except Exception as e:
            test_result['status'] = 'error'
            test_result['details'] = {'error': str(e)}
            logger.error(f"Performance tests: ERROR - {e}")
        
        return test_result
    
    def check_api_availability(self, api_url: str = "http://localhost:5000", timeout: int = 30) -> bool:
        """Kiểm tra API có sẵn sàng không"""
        logger.info(f"Kiểm tra API availability: {api_url}")
        
        try:
            import requests
            response = requests.get(f"{api_url}/health", timeout=timeout)
            if response.status_code == 200:
                logger.info("API is available")
                return True
            else:
                logger.warning(f"API returned status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API is not available: {e}")
            return False
    
    def run_all_tests(self, api_url: str = "http://localhost:5000", 
                     skip_performance: bool = False,
                     image_count: int = 50,
                     performance_duration: int = 60) -> dict:
        """
        Chạy tất cả tests
        
        Args:
            api_url: URL của API
            skip_performance: Bỏ qua performance tests
            image_count: Số lượng ảnh cho performance test
            performance_duration: Thời gian performance test
            
        Returns:
            dict: Kết quả tổng thể
        """
        logger.info("Bắt đầu chạy tất cả tests...")
        
        # Kiểm tra API availability
        if not self.check_api_availability(api_url):
            logger.warning("API không khả dụng, một số tests có thể thất bại")
        
        # Chạy unit tests
        unit_result = self.run_unit_tests()
        self.overall_results['tests_run'].append(unit_result)
        
        # Chạy system tests
        system_result = self.run_system_tests(api_url)
        self.overall_results['tests_run'].append(system_result)
        
        # Chạy performance tests (nếu không bỏ qua)
        if not skip_performance:
            performance_result = self.run_performance_tests(
                api_url, image_count, performance_duration
            )
            self.overall_results['tests_run'].append(performance_result)
        
        # Tính toán summary
        self._calculate_overall_summary()
        
        # Lưu kết quả
        self._save_overall_results()
        
        # In summary
        self._print_summary()
        
        logger.info("Hoàn thành tất cả tests!")
        return self.overall_results
    
    def _calculate_overall_summary(self):
        """Tính toán summary tổng thể"""
        total_tests = len(self.overall_results['tests_run'])
        passed_tests = len([t for t in self.overall_results['tests_run'] if t['status'] == 'passed'])
        failed_tests = total_tests - passed_tests
        
        self.overall_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
    
    def _save_overall_results(self):
        """Lưu kết quả tổng thể"""
        # Lưu kết quả JSON
        result_file = self.result_dir / f"overall_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        import json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.overall_results, f, indent=2, ensure_ascii=False)
        
        # Tạo báo cáo HTML
        self._generate_overall_html_report()
        
        # Tạo báo cáo text
        self._generate_overall_text_report()
        
        logger.info(f"Kết quả tổng thể đã được lưu tại: {result_file}")
    
    def _generate_overall_html_report(self):
        """Tạo báo cáo HTML tổng thể"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition System - Overall Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .test-result {{ margin: 15px 0; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }}
        .passed {{ background-color: #d4edda; border-color: #c3e6cb; }}
        .failed {{ background-color: #f8d7da; border-color: #f5c6cb; }}
        .timeout {{ background-color: #fff3cd; border-color: #ffeaa7; }}
        .error {{ background-color: #f8d7da; border-color: #f5c6cb; }}
        .status-badge {{ 
            display: inline-block; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-weight: bold; 
            font-size: 0.9em; 
        }}
        .status-passed {{ background-color: #28a745; color: white; }}
        .status-failed {{ background-color: #dc3545; color: white; }}
        .status-timeout {{ background-color: #ffc107; color: black; }}
        .status-error {{ background-color: #dc3545; color: white; }}
        .details {{ margin-top: 10px; }}
        .details summary {{ cursor: pointer; font-weight: bold; }}
        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Face Recognition System - Overall Test Report</h1>
        <p><strong>Timestamp:</strong> {self.overall_results['timestamp']}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Total Tests</th>
                <td style="border: 1px solid #ddd; padding: 8px;">{self.overall_results['summary']['total_tests']}</td>
            </tr>
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Passed</th>
                <td style="border: 1px solid #ddd; padding: 8px; color: #28a745; font-weight: bold;">{self.overall_results['summary']['passed']}</td>
            </tr>
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Failed</th>
                <td style="border: 1px solid #ddd; padding: 8px; color: #dc3545; font-weight: bold;">{self.overall_results['summary']['failed']}</td>
            </tr>
            <tr>
                <th style="border: 1px solid #ddd; padding: 8px; text-align: left;">Success Rate</th>
                <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{self.overall_results['summary']['success_rate']:.1f}%</td>
            </tr>
        </table>
    </div>
    
    <h2>Test Results</h2>
"""
        
        for test in self.overall_results['tests_run']:
            status_class = test['status']
            status_badge_class = f"status-{test['status']}"
            
            html_content += f"""
    <div class="test-result {status_class}">
        <h3>{test['test_type'].replace('_', ' ').title()}</h3>
        <span class="status-badge {status_badge_class}">{test['status'].upper()}</span>
        <p><strong>Duration:</strong> {test.get('duration', 0):.2f}s</p>
        <p><strong>Return Code:</strong> {test.get('return_code', 'N/A')}</p>
        
        <div class="details">
            <details>
                <summary>Test Details</summary>
                <h4>STDOUT:</h4>
                <pre>{test.get('stdout', 'No output')}</pre>
                <h4>STDERR:</h4>
                <pre>{test.get('stderr', 'No errors')}</pre>
                <h4>Additional Details:</h4>
                <pre>{test.get('details', {})}</pre>
            </details>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        html_file = self.result_dir / f"overall_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Báo cáo tổng thể HTML đã được tạo: {html_file}")
    
    def _generate_overall_text_report(self):
        """Tạo báo cáo text tổng thể"""
        text_content = f"""
FACE RECOGNITION SYSTEM - OVERALL TEST REPORT
=============================================

Timestamp: {self.overall_results['timestamp']}

TEST SUMMARY
-----------
Total Tests: {self.overall_results['summary']['total_tests']}
Passed: {self.overall_results['summary']['passed']}
Failed: {self.overall_results['summary']['failed']}
Success Rate: {self.overall_results['summary']['success_rate']:.1f}%

DETAILED RESULTS
---------------
"""
        
        for test in self.overall_results['tests_run']:
            text_content += f"""
Test Type: {test['test_type']}
Status: {test['status'].upper()}
Duration: {test.get('duration', 0):.2f}s
Return Code: {test.get('return_code', 'N/A')}

STDOUT:
{test.get('stdout', 'No output')}

STDERR:
{test.get('stderr', 'No errors')}

Additional Details:
{test.get('details', {})}

{'='*50}
"""
        
        text_file = self.result_dir / f"overall_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"Báo cáo tổng thể text đã được tạo: {text_file}")
    
    def _print_summary(self):
        """In summary ra console"""
        print("\n" + "="*70)
        print("OVERALL TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {self.overall_results['summary']['total_tests']}")
        print(f"Passed: {self.overall_results['summary']['passed']}")
        print(f"Failed: {self.overall_results['summary']['failed']}")
        print(f"Success Rate: {self.overall_results['summary']['success_rate']:.1f}%")
        print("="*70)
        
        print("\nDETAILED RESULTS:")
        for test in self.overall_results['tests_run']:
            status_icon = "✅" if test['status'] == 'passed' else "❌"
            print(f"{status_icon} {test['test_type']}: {test['status'].upper()} ({test.get('duration', 0):.2f}s)")
        
        if self.overall_results['summary']['failed'] > 0:
            print("\nFAILED TESTS:")
            for test in self.overall_results['tests_run']:
                if test['status'] != 'passed':
                    print(f"- {test['test_type']}: {test.get('details', {}).get('error', 'Unknown error')}")
        
        print(f"\nKết quả chi tiết đã được lưu tại: {self.result_dir}")

def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description='Chạy tất cả tests hệ thống nhận diện khuôn mặt')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                       help='URL của Flask API (default: http://localhost:5000)')
    parser.add_argument('--skip-performance', action='store_true',
                       help='Bỏ qua performance tests')
    parser.add_argument('--images', type=int, default=50,
                       help='Số lượng ảnh cho performance test (default: 50)')
    parser.add_argument('--duration', type=int, default=60,
                       help='Thời gian performance test (giây, default: 60)')
    parser.add_argument('--unit-only', action='store_true',
                       help='Chỉ chạy unit tests')
    parser.add_argument('--system-only', action='store_true',
                       help='Chỉ chạy system tests')
    parser.add_argument('--performance-only', action='store_true',
                       help='Chỉ chạy performance tests')
    
    args = parser.parse_args()
    
    # Khởi tạo test runner
    runner = TestRunner()
    
    try:
        if args.unit_only:
            # Chỉ chạy unit tests
            result = runner.run_unit_tests()
            runner.overall_results['tests_run'].append(result)
        elif args.system_only:
            # Chỉ chạy system tests
            result = runner.run_system_tests(args.url)
            runner.overall_results['tests_run'].append(result)
        elif args.performance_only:
            # Chỉ chạy performance tests
            result = runner.run_performance_tests(args.url, args.images, args.duration)
            runner.overall_results['tests_run'].append(result)
        else:
            # Chạy tất cả tests
            runner.run_all_tests(
                api_url=args.url,
                skip_performance=args.skip_performance,
                image_count=args.images,
                performance_duration=args.duration
            )
        
        # Tính toán và lưu kết quả
        runner._calculate_overall_summary()
        runner._save_overall_results()
        runner._print_summary()
        
    except KeyboardInterrupt:
        print("\nTests bị gián đoạn bởi người dùng")
    except Exception as e:
        print(f"\nLỗi khi chạy tests: {e}")
        logger.error(f"Lỗi khi chạy tests: {e}")

if __name__ == '__main__':
    main() 