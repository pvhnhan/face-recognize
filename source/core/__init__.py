"""
Core module cho hệ thống nhận diện khuôn mặt
Chứa các script training và inference chính
"""

from .train import FaceRecognitionTrainer
from .inference import FaceRecognitionInference

__all__ = [
    'FaceRecognitionTrainer',
    'FaceRecognitionInference'
] 