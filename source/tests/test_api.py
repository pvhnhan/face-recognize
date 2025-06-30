#!/usr/bin/env python3
"""
Script test API nhận diện khuôn mặt
"""

import requests
import json
import base64
import os
from pathlib import Path

# Cấu hình
API_BASE_URL = "http://localhost:5000"
TEST_IMAGE_PATH = "data/raw_images/test.jpg"  # Đường dẫn ảnh test

def test_health_check():
    """Test endpoint health check"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_docs():
    """Test endpoint API docs"""
    print("\n📚 Testing API docs...")
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        print(f"Status: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def encode_image_to_base64(image_path):
    """Encode ảnh thành base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file ảnh: {image_path}")
        return None

def test_face_recognition_single():
    """Test API nhận diện khuôn mặt đơn lẻ"""
    print("\n👤 Testing face recognition (single image)...")
    
    # Kiểm tra file ảnh test
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"⚠️  Không tìm thấy ảnh test tại: {TEST_IMAGE_PATH}")
        print("   Hãy đặt một ảnh test vào thư mục data/raw_images/")
        return False
    
    # Encode ảnh thành base64
    image_base64 = encode_image_to_base64(TEST_IMAGE_PATH)
    if not image_base64:
        return False
    
    # Chuẩn bị data
    data = {
        "image": image_base64,
        "threshold": 0.6
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/face-recognition",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ API response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Error response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_face_recognition_batch():
    """Test API nhận diện khuôn mặt batch"""
    print("\n👥 Testing face recognition (batch)...")
    
    # Tìm tất cả ảnh trong thư mục test
    test_dir = Path("data/raw_images")
    if not test_dir.exists():
        print(f"⚠️  Không tìm thấy thư mục: {test_dir}")
        return False
    
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not image_files:
        print("⚠️  Không tìm thấy ảnh nào trong thư mục test")
        return False
    
    # Lấy 3 ảnh đầu tiên để test
    test_images = image_files[:3]
    images_base64 = []
    
    for img_path in test_images:
        img_base64 = encode_image_to_base64(str(img_path))
        if img_base64:
            images_base64.append(img_base64)
    
    if not images_base64:
        print("❌ Không thể encode được ảnh nào")
        return False
    
    # Chuẩn bị data
    data = {
        "images": images_base64,
        "threshold": 0.6
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/face-recognition/batch",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ API response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Error response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_statistics():
    """Test API thống kê"""
    print("\n📊 Testing statistics...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/statistics")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ API response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Error response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Chạy tất cả test"""
    print("🚀 Bắt đầu test API nhận diện khuôn mặt")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("API Docs", test_api_docs),
        ("Statistics", test_statistics),
        ("Face Recognition (Single)", test_face_recognition_single),
        ("Face Recognition (Batch)", test_face_recognition_batch),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # Tổng kết
    print("\n" + "="*50)
    print("📋 KẾT QUẢ TEST:")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nTổng cộng: {passed}/{len(results)} test thành công")
    
    if passed == len(results):
        print("🎉 Tất cả test đều thành công!")
    else:
        print("⚠️  Một số test thất bại. Hãy kiểm tra lại.")

if __name__ == "__main__":
    main() 