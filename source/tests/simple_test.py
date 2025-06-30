#!/usr/bin/env python3
"""
Script test đơn giản để kiểm tra API nhận diện khuôn mặt
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime

def test_health_endpoint(base_url):
    """Test health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_docs_endpoint(base_url):
    """Test docs endpoint"""
    print("\nTesting docs endpoint...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Title: {data.get('title')}")
            print(f"Version: {data.get('version')}")
            print(f"Endpoints: {len(data.get('endpoints', {}))}")
            return True
        else:
            print(f"Docs check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error testing docs endpoint: {e}")
        return False

def test_face_recognition_endpoint(base_url):
    """Test face recognition endpoint"""
    print("\nTesting face recognition endpoint...")
    
    # Tạo ảnh test đơn giản
    import numpy as np
    import cv2
    
    # Tạo ảnh test
    image = np.ones((400, 300, 3), dtype=np.uint8) * 255
    
    # Vẽ khuôn mặt giả
    center = (150, 200)
    radius = 50
    cv2.circle(image, center, radius, (200, 200, 200), -1)
    
    # Vẽ mắt
    eye_radius = radius // 4
    left_eye = (center[0] - radius // 3, center[1] - radius // 4)
    right_eye = (center[0] + radius // 3, center[1] - radius // 4)
    cv2.circle(image, left_eye, eye_radius, (0, 0, 0), -1)
    cv2.circle(image, right_eye, eye_radius, (0, 0, 0), -1)
    
    # Vẽ miệng
    mouth_start = (center[0] - radius // 3, center[1] + radius // 3)
    mouth_end = (center[0] + radius // 3, center[1] + radius // 3)
    cv2.line(image, mouth_start, mouth_end, (0, 0, 0), 2)
    
    # Lưu ảnh tạm
    test_image_path = "test_image.jpg"
    cv2.imwrite(test_image_path, image)
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': ('test_image.jpg', f, 'image/jpeg')}
            response = requests.post(
                f"{base_url}/api/face-recognition",
                files=files,
                timeout=30
            )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"Face recognition failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"Error testing face recognition: {e}")
        return False
    
    finally:
        # Xóa ảnh tạm
        if Path(test_image_path).exists():
            Path(test_image_path).unlink()

def test_stats_endpoint(base_url):
    """Test stats endpoint"""
    print("\nTesting stats endpoint...")
    try:
        response = requests.get(f"{base_url}/api/face-recognition/stats", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"Stats check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error testing stats endpoint: {e}")
        return False

def main():
    base_url = "http://localhost:5000"
    
    print("="*50)
    print("SIMPLE API TEST")
    print("="*50)
    print(f"Testing API at: {base_url}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*50)
    
    results = []
    
    # Test health endpoint
    results.append(("Health", test_health_endpoint(base_url)))
    
    # Test docs endpoint
    results.append(("Docs", test_docs_endpoint(base_url)))
    
    # Test face recognition endpoint
    results.append(("Face Recognition", test_face_recognition_endpoint(base_url)))
    
    # Test stats endpoint
    results.append(("Stats", test_stats_endpoint(base_url)))
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    print(f"Success Rate: {(passed / len(results) * 100):.1f}%")
    print("="*50)

if __name__ == '__main__':
    main() 