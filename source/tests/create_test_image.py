#!/usr/bin/env python3
"""
Script tạo ảnh test cho hệ thống nhận diện khuôn mặt
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_simple_face_image():
    """Tạo ảnh khuôn mặt đơn giản bằng OpenCV"""
    # Tạo ảnh trắng
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Vẽ khuôn mặt đơn giản
    # Vẽ hình tròn cho đầu
    cv2.circle(img, (200, 200), 100, (0, 0, 0), 2)
    
    # Vẽ mắt
    cv2.circle(img, (170, 180), 15, (0, 0, 0), -1)
    cv2.circle(img, (230, 180), 15, (0, 0, 0), -1)
    
    # Vẽ mũi
    cv2.line(img, (200, 200), (200, 220), (0, 0, 0), 3)
    
    # Vẽ miệng
    cv2.ellipse(img, (200, 240), (30, 10), 0, 0, 180, (0, 0, 0), 2)
    
    return img

def create_multiple_faces_image():
    """Tạo ảnh có nhiều khuôn mặt"""
    # Tạo ảnh trắng lớn hơn
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Vẽ khuôn mặt 1
    cv2.circle(img, (200, 200), 80, (0, 0, 0), 2)
    cv2.circle(img, (170, 180), 12, (0, 0, 0), -1)
    cv2.circle(img, (230, 180), 12, (0, 0, 0), -1)
    cv2.line(img, (200, 200), (200, 215), (0, 0, 0), 2)
    cv2.ellipse(img, (200, 230), (25, 8), 0, 0, 180, (0, 0, 0), 2)
    
    # Vẽ khuôn mặt 2
    cv2.circle(img, (500, 200), 80, (0, 0, 0), 2)
    cv2.circle(img, (470, 180), 12, (0, 0, 0), -1)
    cv2.circle(img, (530, 180), 12, (0, 0, 0), -1)
    cv2.line(img, (500, 200), (500, 215), (0, 0, 0), 2)
    cv2.ellipse(img, (500, 230), (25, 8), 0, 0, 180, (0, 0, 0), 2)
    
    # Vẽ khuôn mặt 3
    cv2.circle(img, (350, 400), 80, (0, 0, 0), 2)
    cv2.circle(img, (320, 380), 12, (0, 0, 0), -1)
    cv2.circle(img, (380, 380), 12, (0, 0, 0), -1)
    cv2.line(img, (350, 400), (350, 415), (0, 0, 0), 2)
    cv2.ellipse(img, (350, 430), (25, 8), 0, 0, 180, (0, 0, 0), 2)
    
    return img

def create_realistic_face_image():
    """Tạo ảnh khuôn mặt thực tế hơn"""
    # Tạo ảnh với màu da
    img = np.ones((400, 400, 3), dtype=np.uint8)
    img[:, :, 0] = 200  # Blue
    img[:, :, 1] = 180  # Green  
    img[:, :, 2] = 160  # Red (màu da)
    
    # Tạo mask cho khuôn mặt
    mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(mask, (200, 200), 120, 255, -1)
    
    # Áp dụng mask
    for i in range(3):
        img[:, :, i] = cv2.bitwise_and(img[:, :, i], mask)
    
    # Vẽ các chi tiết khuôn mặt
    # Mắt
    cv2.ellipse(img, (170, 180), (20, 12), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, (230, 180), (20, 12), 0, 0, 360, (50, 50, 50), -1)
    
    # Mũi
    cv2.ellipse(img, (200, 210), (8, 15), 0, 0, 360, (100, 100, 100), -1)
    
    # Miệng
    cv2.ellipse(img, (200, 250), (30, 15), 0, 0, 180, (80, 80, 80), -1)
    
    # Tóc
    cv2.ellipse(img, (200, 120), (80, 40), 0, 0, 180, (30, 30, 30), -1)
    
    return img

def main():
    """Tạo các ảnh test"""
    print("🎨 Tạo ảnh test cho hệ thống nhận diện khuôn mặt...")
    
    # Tạo thư mục data/raw_images nếu chưa có
    output_dir = Path("data/raw_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo ảnh khuôn mặt đơn giản
    print("📸 Tạo ảnh khuôn mặt đơn giản...")
    simple_face = create_simple_face_image()
    cv2.imwrite(str(output_dir / "test_simple_face.jpg"), simple_face)
    
    # Tạo ảnh nhiều khuôn mặt
    print("👥 Tạo ảnh nhiều khuôn mặt...")
    multiple_faces = create_multiple_faces_image()
    cv2.imwrite(str(output_dir / "test_multiple_faces.jpg"), multiple_faces)
    
    # Tạo ảnh khuôn mặt thực tế
    print("🎭 Tạo ảnh khuôn mặt thực tế...")
    realistic_face = create_realistic_face_image()
    cv2.imwrite(str(output_dir / "test_realistic_face.jpg"), realistic_face)
    
    # Tạo ảnh test chính
    print("✅ Tạo ảnh test chính...")
    cv2.imwrite(str(output_dir / "test.jpg"), simple_face)
    
    print(f"✅ Đã tạo 4 ảnh test trong thư mục: {output_dir}")
    print("📁 Các file đã tạo:")
    for img_file in output_dir.glob("*.jpg"):
        print(f"   - {img_file.name}")

if __name__ == "__main__":
    main() 