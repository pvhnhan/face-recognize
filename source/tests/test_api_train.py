#!/usr/bin/env python3
"""
Script test API train đơn giản
"""

import requests
import time
import json

def test_train_api():
    """Test API train"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing API Train...")
    
    # Test 1: Bắt đầu training
    print("\n1. Bắt đầu training...")
    try:
        response = requests.post(f"{base_url}/api/face-recognition/train")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Training started successfully")
        elif response.status_code == 409:
            print("⚠️ Training already running")
        else:
            print("❌ Failed to start training")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Kiểm tra status
    print("\n2. Kiểm tra training status...")
    try:
        response = requests.get(f"{base_url}/api/face-recognition/train/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Status check successful")
        else:
            print("❌ Failed to get status")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 3: Chờ và kiểm tra kết quả
    print("\n3. Chờ training hoàn thành...")
    max_wait = 300  # 5 phút
    wait_time = 0
    
    while wait_time < max_wait:
        try:
            response = requests.get(f"{base_url}/api/face-recognition/train/status")
            if response.status_code == 200:
                status_data = response.json()
                
                if not status_data.get('running', False):
                    print("✅ Training completed!")
                    print(f"Success: {status_data.get('success', False)}")
                    print(f"Log: {status_data.get('log', 'No log')}")
                    return status_data.get('success', False)
                
                elapsed = status_data.get('elapsed_time', 0)
                print(f"⏳ Training running... ({elapsed:.1f}s elapsed)")
                
            time.sleep(10)
            wait_time += 10
            
        except Exception as e:
            print(f"❌ Error checking status: {e}")
            time.sleep(10)
            wait_time += 10
    
    print("⏰ Timeout waiting for training")
    return False

def test_system_status():
    """Test system status"""
    base_url = "http://localhost:5000"
    
    print("\n🧪 Testing System Status...")
    
    try:
        response = requests.get(f"{base_url}/api/face-recognition/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ System status check successful")
            return True
        else:
            print("❌ Failed to get system status")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 API Train Test")
    print("=" * 50)
    
    # Test system status first
    if not test_system_status():
        print("❌ System not ready")
        return
    
    # Test train API
    success = test_train_api()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")

if __name__ == "__main__":
    main() 