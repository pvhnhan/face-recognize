# 🐳 Hướng dẫn sử dụng Docker Compose

Hướng dẫn chi tiết cách chạy hệ thống nhận diện khuôn mặt với Docker Compose.

## 📋 Yêu cầu trước khi chạy

### 1. Cài đặt Docker
- **Windows**: Tải và cài đặt [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: Cài đặt Docker Engine và Docker Compose
- **macOS**: Tải và cài đặt [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 2. Kiểm tra Docker
```bash
# Kiểm tra Docker version
docker --version
docker-compose --version

# Kiểm tra Docker daemon
docker info
```

### 3. Cài đặt NVIDIA Docker (cho GPU)
Nếu bạn muốn sử dụng GPU:
```bash
# Cài đặt NVIDIA Container Toolkit
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Kiểm tra
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 Chạy hệ thống với Docker Compose

### 1. Chuẩn bị dữ liệu
Trước khi chạy, hãy đảm bảo có cấu trúc thư mục:
```
source/
├── data/
│   ├── raw_images/          # Ảnh gốc
│   ├── yolo_dataset/        # Dataset cho training
│   └── metadata.csv         # Metadata
├── models/
│   ├── pretrained/          # Pretrained weights
│   └── trained/             # Trained models
└── docker/
    ├── docker-compose.yml
    ├── Dockerfile.cpu
    └── Dockerfile.gpu
```

### 2. Chạy với CPU (khuyến nghị cho bắt đầu)

#### Build và chạy:
```bash
cd docker
docker-compose --profile cpu up --build
```

#### Chạy nền:
```bash
docker-compose --profile cpu up --build -d
```

#### Xem logs:
```bash
docker-compose --profile cpu logs -f
```

### 3. Chạy với GPU (nếu có NVIDIA GPU)

#### Build và chạy:
```bash
cd docker
docker-compose --profile gpu up --build
```

#### Chạy nền:
```bash
docker-compose --profile gpu up --build -d
```

#### Xem logs:
```bash
docker-compose --profile gpu logs -f
```

### 4. Chạy training với GPU

```bash
cd docker
docker-compose --profile training up --build
```

## 🛠️ Các lệnh Docker Compose hữu ích

### Quản lý containers
```bash
# Dừng tất cả services
docker-compose down

# Dừng và xóa volumes
docker-compose down -v

# Restart service
docker-compose restart

# Xem trạng thái
docker-compose ps

# Xem logs real-time
docker-compose logs -f

# Xem logs của service cụ thể
docker-compose logs -f face-recognition-cpu
```

### Quản lý images
```bash
# Xóa tất cả images
docker-compose down --rmi all

# Xóa theo profile
docker-compose --profile cpu-dev down --rmi local

# Build lại images
docker-compose build --no-cache

# Build theo profile
docker-compose --profile cpu-dev up --build -d

# Pull latest images
docker-compose pull
```

# Xem resource usage
docker stats

# Xem disk usage
docker system df
```

## 🌐 Truy cập API

Sau khi chạy thành công, API sẽ có sẵn tại:

- **URL**: http://localhost:5000
- **Health Check**: http://localhost:5000/health
- **API Docs**: http://localhost:5000/docs

### Test API
```bash
# Health check
curl http://localhost:5000/health

# Face recognition
curl -X POST -F "image=@test.jpg" http://localhost:5000/api/face-recognition/recognize

# System status
curl http://localhost:5000/api/face-recognition/status
```

## ⚙️ Cấu hình nâng cao

### 1. Thay đổi port
Chỉnh sửa `docker-compose.yml`:
```yaml
ports:
  - "8080:5000"  # Thay đổi từ 5000 sang 8080
```

### 2. Thêm environment variables
```yaml
environment:
  - DEVICE=cpu
  - FLASK_ENV=development
  - DEBUG=True
  - LOG_LEVEL=DEBUG
```

### 3. Mount thêm volumes
```yaml
volumes:
  - ../data:/app/data
  - ../models:/app/models
  - ../config:/app/config  # Thêm config
  - ../outputs:/app/outputs  # Thêm outputs
```

### 4. Sử dụng GPU khác
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=1  # Sử dụng GPU thứ 2
```

## 🔧 Troubleshooting

### 1. Lỗi "port already in use"
```bash
# Tìm process đang sử dụng port 5000
netstat -ano | findstr :5000  # Windows
lsof -i :5000                 # Linux/macOS

# Kill process
kill -9 <PID>
```

### 2. Lỗi GPU không được nhận diện
```bash
# Kiểm tra NVIDIA drivers
nvidia-smi

# Kiểm tra Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 3. Lỗi permission
```bash
# Windows: Chạy Docker Desktop với quyền admin
# Linux: Thêm user vào docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 4. Lỗi memory
```bash
# Tăng memory limit trong Docker Desktop
# Hoặc sử dụng swap
```

### 5. Lỗi build
```bash
# Clean và build lại
docker-compose down --rmi all
docker system prune -f
docker-compose --profile cpu up --build
```

## 📊 Monitoring

### 1. Xem resource usage
```bash
# Real-time stats
docker stats

# Container logs
docker-compose logs -f
```

### 2. Health check
```bash
# API health
curl http://localhost:5000/health

# Container health
docker-compose ps
```

## 🧹 Cleanup

### 1. Dừng và xóa containers
```bash
docker-compose down
```

### 2. Xóa images
```bash
docker-compose down --rmi all
```

### 3. Xóa volumes
```bash
docker-compose down -v
```

### 4. Cleanup toàn bộ
```bash
docker system prune -a -f
docker volume prune -f
```

## 🎯 Best Practices

1. **Luôn sử dụng `--profile`** để chọn chế độ phù hợp
2. **Mount volumes** để dữ liệu không bị mất khi container restart
3. **Sử dụng `.env` file** cho environment variables
4. **Monitor logs** để debug khi có lỗi
5. **Backup data** trước khi cleanup
6. **Sử dụng GPU** cho training, CPU cho inference nhẹ 