# 📁 Cấu trúc Project - Face Recognition System

## 🎯 Tổng quan

Project đã được clean và tổ chức lại với cấu trúc rõ ràng, dễ bảo trì và mở rộng.

## 📂 Cấu trúc thư mục

```
source/
├── 📁 api/                    # Flask API endpoints
│   ├── __init__.py
│   └── face_recognition.py    # Face recognition API
│
├── 📁 config/                 # Cấu hình hệ thống
│   ├── __init__.py
│   ├── config.py             # File cấu hình chính
│   └── yolo/                 # YOLO configuration files
│
├── 📁 core/                   # Core modules chính
│   ├── __init__.py           # Core module init
│   ├── train.py              # Training script
│   └── inference.py          # Inference script
│
├── 📁 data/                   # Dữ liệu và dataset
│   ├── raw_images/           # Ảnh gốc từ internet
│   │   └── .gitkeep
│   ├── yolo_dataset/         # Dataset cho YOLO training
│   │   └── .gitkeep
│   └── metadata.csv          # Metadata của dataset
│
├── 📁 docker/                 # Docker configuration
│   ├── Dockerfile.cpu        # Dockerfile cho CPU
│   ├── Dockerfile.gpu        # Dockerfile cho GPU
│   ├── mongo-init.js         # MongoDB initialization
│   └── README_DOCKER.md      # Docker documentation
│
├── 📁 logs/                   # Log files
│   ├── .gitkeep
│   └── app.log               # Application logs
│
├── 📁 models/                 # Models và weights
│   ├── face_embeddings/      # Face embeddings
│   │   └── .gitkeep
│   ├── pretrained/           # Pretrained models
│   │   └── .gitkeep
│   └── trained/              # Trained models
│       └── .gitkeep
│
├── 📁 tests/                  # Test scripts
│   ├── __init__.py           # Tests module init
│   ├── test_api.py           # API test script
│   └── create_test_image.py  # Test image generator
│
├── 📁 utils/                  # Utility modules
│   ├── __init__.py
│   ├── data_processor.py     # Data processing utilities
│   ├── database.py           # Database utilities
│   ├── face_utils.py         # Face processing utilities
│   └── image_utils.py        # Image processing utilities
│
├── 📁 yolov7/                 # YOLOv7 repository (cloned)
│
├── 📁 outputs/                # Inference results
│   └── .gitkeep
│
├── 📁 temp/                   # Temporary files
│   └── .gitkeep
│
├── 📄 app.py                  # Flask application chính
├── 📄 docker-compose.yml      # Docker Compose configuration
├── 📄 requirements.txt        # Python dependencies
├── 📄 build.sh               # Build script (Linux/macOS)
├── 📄 build.bat              # Build script (Windows)
├── 📄 .gitignore             # Git ignore rules
├── 📄 .dockerignore          # Docker ignore rules
├── 📄 README.md              # Project documentation
└── 📄 PROJECT_STRUCTURE.md   # File này
```

## 🔧 Các file quan trọng

### Core Files
- **`app.py`**: Flask application chính
- **`core/train.py`**: Script training pipeline
- **`core/inference.py`**: Script inference
- **`api/face_recognition.py`**: Face recognition API endpoints

### Configuration Files
- **`config/config.py`**: Cấu hình hệ thống
- **`requirements.txt`**: Python dependencies
- **`docker-compose.yml`**: Docker Compose configuration
- **`.env.example`**: Template cho environment variables

### Build Scripts
- **`build.sh`**: Build script cho Linux/macOS
- **`build.bat`**: Build script cho Windows

### Test Files
- **`tests/test_api.py`**: API test script
- **`tests/create_test_image.py`**: Test image generator

## 🚀 Workflow sử dụng

### 1. Setup môi trường
```bash
# Linux/macOS
./build.sh setup

# Windows
build.bat setup
```

### 2. Chạy application
```bash
# Linux/macOS
./build.sh run

# Windows
build.bat run
```

### 3. Training
```bash
# Linux/macOS
./build.sh train

# Windows
build.bat train
```

### 4. Testing
```bash
# Linux/macOS
./build.sh test

# Windows
build.bat test
```

### 5. Docker
```bash
# Build images
./build.sh docker-build cpu
./build.sh docker-build gpu

# Run containers
./build.sh docker-run cpu
./build.sh docker-run gpu
```

## 📊 Cấu trúc dữ liệu

### Data Flow
1. **Raw Images** → `data/raw_images/`
2. **Metadata** → `data/metadata.csv`
3. **YOLO Dataset** → `data/yolo_dataset/`
4. **Face Embeddings** → `models/face_embeddings/`
5. **Trained Models** → `models/trained/`
6. **Inference Results** → `outputs/`

### Model Pipeline
1. **Data Collection** → Raw images + metadata
2. **Data Processing** → YOLO format conversion
3. **Training** → YOLOv7 face detection
4. **Embedding Creation** → Face embeddings extraction
5. **Inference** → Face detection + recognition

## 🔒 Security & Best Practices

### Git Management
- **`.gitignore`**: Ignore sensitive files và build artifacts
- **`.gitkeep`**: Giữ cấu trúc thư mục trống
- **No sensitive data**: Không commit models, data, logs

### Environment Management
- **Virtual Environment**: Isolated Python environment
- **Environment Variables**: Sensitive configs in `.env`
- **Docker**: Containerized deployment

### Code Organization
- **Modular Design**: Separate concerns into modules
- **Clear Naming**: Descriptive file and folder names
- **Documentation**: Comprehensive README và comments

## 🎯 Lợi ích của cấu trúc mới

### ✅ Clean & Organized
- Tách biệt rõ ràng các thành phần
- Dễ tìm và sửa đổi code
- Cấu trúc nhất quán

### ✅ Scalable
- Dễ thêm tính năng mới
- Modular design cho phép mở rộng
- Clear separation of concerns

### ✅ Maintainable
- Code được tổ chức logic
- Dễ debug và test
- Documentation đầy đủ

### ✅ Cross-platform
- Hỗ trợ cả Linux/macOS và Windows
- Build scripts cho từng platform
- Docker support

### ✅ Production Ready
- Docker containerization
- Environment management
- Security best practices
- Comprehensive testing

## 🔄 Migration Notes

### Files đã được di chuyển:
- `train.py` → `core/train.py`
- `inference.py` → `core/inference.py`
- `test_api.py` → `tests/test_api.py`
- `create_test_image.py` → `tests/create_test_image.py`

### Files đã được tạo:
- `core/__init__.py`
- `tests/__init__.py`
- `build.bat` (Windows support)
- `.gitkeep` files cho empty directories
- `PROJECT_STRUCTURE.md`

### Files đã được cập nhật:
- `README.md` với cấu trúc mới
- `.gitignore` với rules phù hợp
- Import paths trong các modules

---

**🎉 Project đã sẵn sàng cho development và production!** 