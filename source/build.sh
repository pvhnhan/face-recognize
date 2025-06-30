#!/bin/bash

# Script build và setup hệ thống nhận diện khuôn mặt
# Sử dụng YOLOv7 chính thức từ repository GitHub: https://github.com/WongKinYiu/yolov7

set -e  # Dừng script nếu có lỗi

# Màu sắc cho output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thông tin dự án
PROJECT_NAME="Face Recognition System"
VERSION="1.0.0"
AUTHOR="AI Assistant"

# Đường dẫn
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Cấu hình
PYTHON_VERSION="3.8"
VENV_NAME="venv"
DOCKER_CPU_IMAGE="face-recognition-cpu:latest"
DOCKER_GPU_IMAGE="face-recognition-gpu:latest"
CONTAINER_NAME="face-recognition-api"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "=========================================="
    echo "  $PROJECT_NAME v$VERSION"
    echo "  $AUTHOR"
    echo "=========================================="
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}[STEP] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Function để hiển thị lỗi và chờ người dùng
show_error_and_wait() {
    local error_msg="$1"
    local exit_code="${2:-1}"
    
    echo ""
    echo -e "${RED}=========================================="
    echo -e "  LỖI XẢY RA!"
    echo -e "==========================================${NC}"
    echo -e "${RED}$error_msg${NC}"
    echo ""
    echo -e "${YELLOW}Nhấn Enter để tiếp tục hoặc Ctrl+C để thoát...${NC}"
    read -r
    exit $exit_code
}

# Function để hiển thị cảnh báo và chờ người dùng
show_warning_and_wait() {
    local warning_msg="$1"
    
    echo ""
    echo -e "${YELLOW}=========================================="
    echo -e "  CẢNH BÁO!"
    echo -e "==========================================${NC}"
    echo -e "${YELLOW}$warning_msg${NC}"
    echo ""
    echo -e "${YELLOW}Nhấn Enter để tiếp tục hoặc Ctrl+C để thoát...${NC}"
    read -r
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        show_error_and_wait "$1 không được cài đặt. Vui lòng cài đặt $1 trước khi tiếp tục."
        return 1
    fi
    return 0
}

check_python_version() {
    print_info "Kiểm tra Python version..."
    
    # Kiểm tra Python trên Windows và Linux
    PYTHON_CMD=""
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        show_error_and_wait "Python không được cài đặt. Vui lòng cài đặt Python 3.8 trở lên."
        return 1
    fi
    
    # Kiểm tra version
    PYTHON_VERSION_ACTUAL=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python $PYTHON_VERSION_ACTUAL được tìm thấy (sử dụng: $PYTHON_CMD)"
        # Lưu command để sử dụng sau này
        export PYTHON_CMD="$PYTHON_CMD"
        return 0
    else
        show_error_and_wait "Cần Python $PYTHON_VERSION trở lên, hiện tại: $PYTHON_VERSION_ACTUAL. Vui lòng nâng cấp Python."
        return 1
    fi
}

check_docker() {
    print_info "Kiểm tra Docker..."
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_info "Docker version: $DOCKER_VERSION"
        
        # Kiểm tra Docker daemon
        if docker info &> /dev/null; then
            print_success "Docker daemon đang chạy"
        else
            show_warning_and_wait "Docker daemon không chạy. Vui lòng chạy lệnh: sudo systemctl start docker (Linux) hoặc khởi động Docker Desktop (Windows/Mac)"
        fi
    else
        show_warning_and_wait "Docker không được cài đặt. Một số tính năng có thể không hoạt động."
    fi
}

check_cuda() {
    print_info "Kiểm tra CUDA..."
    
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        print_info "NVIDIA Driver version: $CUDA_VERSION"
        
        # Kiểm tra PyTorch CUDA
        if $PYTHON_CMD -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
            print_success "PyTorch CUDA khả dụng"
        else
            show_warning_and_wait "PyTorch CUDA không khả dụng. Training sẽ chạy trên CPU (chậm hơn)."
        fi
    else
        show_warning_and_wait "NVIDIA GPU không được phát hiện. Training sẽ chạy trên CPU (chậm hơn)."
    fi
}

setup_virtual_environment() {
    print_step "Thiết lập môi trường ảo Python"
    
    cd "$SCRIPT_DIR"
    
    if [ ! -d "$VENV_NAME" ]; then
        print_info "Tạo môi trường ảo..."
        if ! $PYTHON_CMD -m venv "$VENV_NAME"; then
            show_error_and_wait "Lỗi khi tạo môi trường ảo. Kiểm tra quyền ghi và Python installation."
        fi
        print_success "Đã tạo môi trường ảo"
    else
        print_info "Môi trường ảo đã tồn tại"
    fi
    
    # Activate virtual environment
    if [ ! -f "$VENV_NAME/bin/activate" ] && [ ! -f "$VENV_NAME/Scripts/activate" ]; then
        show_error_and_wait "Không tìm thấy script activate trong môi trường ảo. Môi trường ảo có thể bị hỏng."
    fi
    
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    print_info "Nâng cấp pip..."
    if ! pip install --upgrade pip; then
        show_error_and_wait "Lỗi khi nâng cấp pip. Kiểm tra kết nối internet và quyền ghi."
    fi
    
    # Install requirements theo thứ tự ưu tiên
    print_info "Cài đặt dependencies..."
    
    # Cài đặt core dependencies trước
    if [ -f "requirements-core.txt" ]; then
        print_info "Cài đặt core dependencies..."
        if ! pip install -r requirements-core.txt; then
            show_error_and_wait "Lỗi khi cài đặt core dependencies. Kiểm tra kết nối internet và quyền ghi."
        fi
        print_success "Core dependencies đã được cài đặt"
    fi
    
    # Cài đặt ML dependencies
    if [ -f "requirements-ml.txt" ]; then
        print_info "Cài đặt ML dependencies..."
        if ! pip install -r requirements-ml.txt; then
            show_error_and_wait "Lỗi khi cài đặt ML dependencies. Kiểm tra kết nối internet và quyền ghi."
        fi
        print_success "ML dependencies đã được cài đặt"
    fi
    
    # Cài đặt dev dependencies (tùy chọn)
    if [ -f "requirements-dev.txt" ]; then
        print_info "Cài đặt dev dependencies..."
        if ! pip install -r requirements-dev.txt; then
            print_warning "Không thể cài đặt dev dependencies (không bắt buộc)"
        else
            print_success "Dev dependencies đã được cài đặt"
        fi
    fi
    
    print_success "Tất cả dependencies đã được cài đặt"
}

setup_yolov7() {
    print_step "Thiết lập YOLOv7 chính thức"
    
    cd "$SCRIPT_DIR"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    YOLOV7_DIR="yolov7"
    
    if [ ! -d "$YOLOV7_DIR" ]; then
        print_info "Clone YOLOv7 repository..."
        if ! git clone https://github.com/WongKinYiu/yolov7.git "$YOLOV7_DIR"; then
            show_error_and_wait "Lỗi khi clone YOLOv7 repository. Kiểm tra kết nối internet và quyền ghi."
        fi
        print_success "Đã clone YOLOv7 repository"
    else
        print_info "YOLOv7 repository đã tồn tại"
    fi
    
    # Kiểm tra các file cần thiết
    cd "$YOLOV7_DIR"
    
    required_files=(
        "train.py"
        "detect.py"
        "models/experimental.py"
        "utils/general.py"
        "utils/augmentations.py"
        "cfg/training/yolov7.yaml"
        "data/hyp.scratch.p5.yaml"
    )
    
    missing_files=()
    for file_path in "${required_files[@]}"; do
        if [ ! -f "$file_path" ]; then
            missing_files+=("$file_path")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        show_error_and_wait "Thiếu các file YOLOv7: ${missing_files[*]}. Repository có thể bị hỏng, vui lòng xóa thư mục yolov7 và chạy lại."
    fi
    
    # Cài đặt YOLOv7 dependencies
    if [ -f "requirements.txt" ]; then
        print_info "Cài đặt YOLOv7 dependencies..."
        if ! pip install -r requirements.txt; then
            show_warning_and_wait "Lỗi khi cài đặt YOLOv7 dependencies. Một số tính năng có thể không hoạt động."
        else
            print_success "YOLOv7 dependencies đã được cài đặt"
        fi
    fi
    
    print_success "YOLOv7 đã được thiết lập"
    cd ..
}

create_directories() {
    print_step "Tạo cấu trúc thư mục"
    
    cd "$SCRIPT_DIR"
    
    # Tạo các thư mục cần thiết
    directories=(
        "data/raw_images"
        "data/yolo_dataset"
        "data/yolo_dataset/images/train"
        "data/yolo_dataset/images/val"
        "data/yolo_dataset/images/test"
        "data/yolo_dataset/labels/train"
        "data/yolo_dataset/labels/val"
        "data/yolo_dataset/labels/test"
        "models/pretrained"
        "models/trained"
        "models/face_embeddings"
        "logs"
        "temp"
        "outputs"
        "config/yolo"
    )
    
    for dir in "${directories[@]}"; do
        if ! mkdir -p "$dir"; then
            show_error_and_wait "Lỗi khi tạo thư mục: $dir. Kiểm tra quyền ghi."
        fi
        print_info "Tạo thư mục: $dir"
    done
    
    print_success "Cấu trúc thư mục đã được tạo"
}

download_pretrained_weights() {
    print_step "Tải pretrained weights"
    
    cd "$SCRIPT_DIR"
    
    WEIGHTS_DIR="models/pretrained"
    
    # Tạo thư mục nếu chưa có
    mkdir -p "$WEIGHTS_DIR"
    
    # Tải YOLOv7 weights
    YOLOV7_WEIGHTS="$WEIGHTS_DIR/yolov7.pt"
    if [ ! -f "$YOLOV7_WEIGHTS" ]; then
        print_info "Tải YOLOv7 pretrained weights..."
        if ! wget -O "$YOLOV7_WEIGHTS" "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"; then
            show_warning_and_wait "Không thể tải YOLOv7 weights. Bạn có thể tải thủ công từ: https://github.com/WongKinYiu/yolov7/releases"
        else
            print_success "Đã tải YOLOv7 weights"
        fi
    else
        print_info "YOLOv7 weights đã tồn tại"
    fi
}

setup_sample_data() {
    print_step "Thiết lập dữ liệu mẫu"
    
    cd "$SCRIPT_DIR"
    
    # Tạo file metadata mẫu
    if [ ! -f "data/metadata.csv" ]; then
        if ! cat > "data/metadata.csv" << EOF
filename,employee_id,full_name
emp001_001.jpg,EMP001,Nguyen Van A
emp002_001.jpg,EMP002,Tran Thi B
emp003_001.jpg,EMP003,Le Van C
EOF
        then
            show_error_and_wait "Lỗi khi tạo file metadata.csv. Kiểm tra quyền ghi."
        fi
        print_info "Đã tạo file metadata mẫu"
    fi
    
    # Tạo file config mẫu
    if [ ! -f "config/yolo/face_detection.yaml" ]; then
        if ! cat > "config/yolo/face_detection.yaml" << EOF
nc: 1
names: ['face']
path: data/yolo_dataset
train: images/train
val: images/val
test: images/test
EOF
        then
            show_error_and_wait "Lỗi khi tạo file config YOLO. Kiểm tra quyền ghi."
        fi
        print_info "Đã tạo file config YOLO mẫu"
    fi
    
    print_success "Dữ liệu mẫu đã được thiết lập"
}

build_docker_images() {
    print_step "Build Docker images"
    
    cd "$SCRIPT_DIR"
    
    # Check Docker
    if ! check_command "docker"; then
        return 1
    fi
    
    # Parse arguments
    BUILD_MODE="${1:-production}"
    DEVICE_TYPE="${2:-cpu}"
    
    print_info "Build mode: $BUILD_MODE"
    print_info "Device type: $DEVICE_TYPE"
    
    # Build CPU images
    if [ "$DEVICE_TYPE" = "cpu" ] || [ "$DEVICE_TYPE" = "all" ]; then
        if [ -f "docker/Dockerfile.cpu" ]; then
            print_info "Build CPU image ($BUILD_MODE)..."
            if ! docker build -f docker/Dockerfile.cpu --target "$BUILD_MODE" -t "$DOCKER_CPU_IMAGE-$BUILD_MODE" .; then
                show_error_and_wait "Lỗi khi build CPU image. Kiểm tra Dockerfile.cpu và quyền Docker."
            fi
            print_success "CPU image ($BUILD_MODE) đã được build"
        else
            show_error_and_wait "Không tìm thấy docker/Dockerfile.cpu. Vui lòng tạo file Dockerfile.cpu."
        fi
    fi
    
    # Build GPU images
    if [ "$DEVICE_TYPE" = "gpu" ] || [ "$DEVICE_TYPE" = "all" ]; then
        if command -v nvidia-smi &> /dev/null && [ -f "docker/Dockerfile.gpu" ]; then
            print_info "Build GPU image ($BUILD_MODE)..."
            if ! docker build -f docker/Dockerfile.gpu --target "$BUILD_MODE" -t "$DOCKER_GPU_IMAGE-$BUILD_MODE" .; then
                show_warning_and_wait "Lỗi khi build GPU image. Sẽ sử dụng CPU image."
            else
                print_success "GPU image ($BUILD_MODE) đã được build"
            fi
        else
            print_info "GPU không khả dụng hoặc không tìm thấy Dockerfile.gpu"
        fi
    fi
}

run_docker_container() {
    print_step "Chạy Docker container"
    
    # Check Docker
    if ! check_command "docker"; then
        return 1
    fi
    
    # Parse arguments
    BUILD_MODE="${1:-production}"
    DEVICE_TYPE="${2:-cpu}"
    
    print_info "Run mode: $BUILD_MODE"
    print_info "Device type: $DEVICE_TYPE"
    
    # Stop existing container
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        print_info "Dừng container hiện tại..."
        docker stop "$CONTAINER_NAME"
        docker rm "$CONTAINER_NAME"
    fi
    
    # Run new container
    print_info "Chạy container mới..."
    
    # Check if GPU is available
    if [ "$DEVICE_TYPE" = "gpu" ] && command -v nvidia-smi &> /dev/null; then
        print_info "Sử dụng GPU image ($BUILD_MODE)..."
        if ! docker run -d \
            --name "$CONTAINER_NAME" \
            --gpus all \
            -p 5000:5000 \
            -v "$SCRIPT_DIR/data:/app/data" \
            -v "$SCRIPT_DIR/models:/app/models" \
            -v "$SCRIPT_DIR/logs:/app/logs" \
            "$DOCKER_GPU_IMAGE-$BUILD_MODE"; then
            show_error_and_wait "Lỗi khi chạy GPU container. Kiểm tra Docker và GPU drivers."
        fi
    else
        print_info "Sử dụng CPU image ($BUILD_MODE)..."
        if ! docker run -d \
            --name "$CONTAINER_NAME" \
            -p 5000:5000 \
            -v "$SCRIPT_DIR/data:/app/data" \
            -v "$SCRIPT_DIR/models:/app/models" \
            -v "$SCRIPT_DIR/logs:/app/logs" \
            "$DOCKER_CPU_IMAGE-$BUILD_MODE"; then
            show_error_and_wait "Lỗi khi chạy CPU container. Kiểm tra Docker và image."
        fi
    fi
    
    print_success "Container đã được chạy"
    print_info "API có thể truy cập tại: http://localhost:5000"
}

# Function khắc phục lỗi MongoDB
fix_mongodb_issue() {
    print_step "Khắc phục lỗi MongoDB TLS handshake timeout..."
    
    echo -e "${YELLOW}Chọn giải pháp khắc phục MongoDB:${NC}"
    echo "1. Chạy script khắc phục tự động"
    echo "2. Sử dụng MongoDB phiên bản ổn định (5.0)"
    echo "3. Chạy không có MongoDB (chỉ API)"
    echo "4. Thử pull MongoDB thủ công"
    echo "5. Bỏ qua và tiếp tục"
    
    read -p "Nhập lựa chọn (1-5): " choice
    
    case $choice in
        1)
            print_info "Chạy script khắc phục tự động..."
            if [ -f "docker/fix-mongodb-issue.sh" ]; then
                chmod +x docker/fix-mongodb-issue.sh
                ./docker/fix-mongodb-issue.sh
            else
                print_error "Không tìm thấy script fix-mongodb-issue.sh"
            fi
            ;;
        2)
            print_info "Sử dụng MongoDB phiên bản ổn định..."
            if [ -f "docker-compose-stable.yml" ]; then
                print_success "Sử dụng docker-compose-stable.yml"
                DOCKER_COMPOSE_FILE="docker-compose-stable.yml"
            else
                print_error "Không tìm thấy docker-compose-stable.yml"
            fi
            ;;
        3)
            print_info "Chạy không có MongoDB..."
            if [ -f "docker-compose-no-mongo.yml" ]; then
                print_success "Sử dụng docker-compose-no-mongo.yml"
                DOCKER_COMPOSE_FILE="docker-compose-no-mongo.yml"
            else
                print_error "Không tìm thấy docker-compose-no-mongo.yml"
            fi
            ;;
        4)
            print_info "Thử pull MongoDB thủ công..."
            docker pull mongo:5.0 || docker pull mongo:4.4 || print_error "Không thể pull MongoDB"
            ;;
        5)
            print_info "Bỏ qua và tiếp tục..."
            ;;
        *)
            print_error "Lựa chọn không hợp lệ"
            fix_mongodb_issue
            ;;
    esac
}

# Function chạy Docker Compose với xử lý lỗi MongoDB
run_docker_compose() {
    local profile="$1"
    local compose_file="${2:-docker-compose.yml}"
    
    print_step "Chạy Docker Compose với profile: $profile"
    
    # Thử chạy docker-compose
    if docker-compose -f "$compose_file" up --profile "$profile" -d; then
        print_success "Docker Compose chạy thành công!"
        return 0
    else
        print_error "Lỗi khi chạy Docker Compose"
        
        # Kiểm tra có phải lỗi MongoDB không
        if docker-compose -f "$compose_file" logs mongodb 2>&1 | grep -q "TLS handshake timeout"; then
            print_warning "Phát hiện lỗi MongoDB TLS handshake timeout"
            fix_mongodb_issue
            # Thử lại sau khi khắc phục
            if [ -n "$DOCKER_COMPOSE_FILE" ]; then
                compose_file="$DOCKER_COMPOSE_FILE"
            fi
            if docker-compose -f "$compose_file" up --profile "$profile" -d; then
                print_success "Docker Compose chạy thành công sau khi khắc phục!"
                return 0
            fi
        fi
        
        return 1
    fi
}

run_flask_app() {
    print_step "Chạy Flask app"
    
    cd "$SCRIPT_DIR"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_NAME" ]; then
        show_error_and_wait "Môi trường ảo chưa được tạo. Chạy '$0 setup' trước."
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Check if app.py exists
    if [ ! -f "app.py" ]; then
        show_error_and_wait "Không tìm thấy app.py. Vui lòng tạo file app.py."
    fi
    
    # Set environment variables
    export FLASK_APP=app.py
    export FLASK_ENV=development
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    
    print_info "Khởi động Flask app..."
    print_info "API có thể truy cập tại: http://localhost:5000"
    print_info "Nhấn Ctrl+C để dừng"
    
    # Run Flask app
    if ! python app.py; then
        show_error_and_wait "Lỗi khi chạy Flask app. Kiểm tra logs để biết chi tiết."
    fi
}

run_training() {
    print_step "Chạy training"
    
    cd "$SCRIPT_DIR"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_NAME" ]; then
        show_error_and_wait "Môi trường ảo chưa được tạo. Chạy '$0 setup' trước."
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Check if train.py exists
    if [ ! -f "train.py" ]; then
        show_error_and_wait "Không tìm thấy train.py. Vui lòng tạo file train.py."
    fi
    
    print_info "Bắt đầu training YOLOv7..."
    
    # Run training
    if ! python train.py; then
        show_error_and_wait "Lỗi khi chạy training. Kiểm tra logs để biết chi tiết."
    fi
    
    print_success "Training hoàn tất"
}

run_inference() {
    print_step "Chạy inference"
    
    cd "$SCRIPT_DIR"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_NAME" ]; then
        show_error_and_wait "Môi trường ảo chưa được tạo. Chạy '$0 setup' trước."
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Check if image path is provided
    if [ -z "$1" ]; then
        show_error_and_wait "Cần cung cấp đường dẫn ảnh. Usage: $0 inference <image_path>"
    fi
    
    # Check if image exists
    if [ ! -f "$1" ]; then
        show_error_and_wait "Không tìm thấy ảnh: $1"
    fi
    
    # Check if inference.py exists
    if [ ! -f "inference.py" ]; then
        show_error_and_wait "Không tìm thấy inference.py. Vui lòng tạo file inference.py."
    fi
    
    print_info "Chạy inference cho ảnh: $1"
    
    # Run inference
    if ! python inference.py --image "$1"; then
        show_error_and_wait "Lỗi khi chạy inference. Kiểm tra logs để biết chi tiết."
    fi
    
    print_success "Inference hoàn tất"
}

run_tests() {
    print_step "Chạy tests"
    
    cd "$SCRIPT_DIR"
    
    # Check if virtual environment exists
    if [ ! -d "$VENV_NAME" ]; then
        show_error_and_wait "Môi trường ảo chưa được tạo. Chạy '$0 setup' trước."
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    print_info "Chạy system tests..."
    
    # Run tests
    if [ -f "test/test_system.py" ]; then
        if ! python test/test_system.py; then
            show_error_and_wait "Tests thất bại. Kiểm tra logs để biết chi tiết."
        fi
    else
        show_warning_and_wait "Không tìm thấy test/test_system.py. Bỏ qua tests."
    fi
    
    print_success "Tests hoàn tất"
}

show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                    Thiết lập môi trường phát triển"
    echo "  docker-build [mode] [device] Build Docker images"
    echo "  docker-run [mode] [device]   Chạy Docker container"
    echo "  docker-compose [mode] [device] Chạy Docker Compose services"
    echo "  run                      Chạy Flask app trong môi trường ảo"
    echo "  train                    Chạy training YOLOv7"
    echo "  inference <img>          Chạy inference cho ảnh"
    echo "  test                     Chạy tests"
    echo "  clean                    Dọn dẹp môi trường"
    echo "  help                     Hiển thị help"
    echo ""
    echo "Options:"
    echo "  mode: production (default) hoặc development"
    echo "  device: cpu (default), gpu, hoặc all"
    echo ""
    echo "Examples:"
    echo "  $0 setup"
    echo "  $0 docker-build production cpu"
    echo "  $0 docker-build development gpu"
    echo "  $0 docker-run production cpu"
    echo "  $0 docker-compose development gpu"
    echo "  $0 run"
    echo "  $0 inference data/test.jpg"
}

clean_environment() {
    print_step "Dọn dẹp môi trường"
    
    cd "$SCRIPT_DIR"
    
    # Remove virtual environment
    if [ -d "$VENV_NAME" ]; then
        print_info "Xóa môi trường ảo..."
        rm -rf "$VENV_NAME"
    fi
    
    # Remove Docker containers and images
    if command -v docker &> /dev/null; then
        print_info "Xóa Docker containers..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        
        print_info "Xóa Docker images..."
        docker rmi "$DOCKER_CPU_IMAGE-production" 2>/dev/null || true
        docker rmi "$DOCKER_CPU_IMAGE-development" 2>/dev/null || true
        docker rmi "$DOCKER_GPU_IMAGE-production" 2>/dev/null || true
        docker rmi "$DOCKER_GPU_IMAGE-development" 2>/dev/null || true
    fi
    
    # Remove logs and temp files
    print_info "Xóa logs và temp files..."
    rm -rf logs/* temp/* outputs/*
    
    print_success "Môi trường đã được dọn dẹp"
}

# Main script
main() {
    print_header
    
    # Check Python version
    if ! check_python_version; then
        exit 1
    fi
    
    # Check Docker
    check_docker
    
    # Check CUDA
    check_cuda
    
    # Parse command
    case "${1:-help}" in
        "setup")
            setup_virtual_environment
            setup_yolov7
            create_directories
            download_pretrained_weights
            setup_sample_data
            print_success "Thiết lập hoàn tất!"
            ;;
        "docker-build")
            build_docker_images "$2" "$3"
            ;;
        "docker-run")
            run_docker_container "$2" "$3"
            ;;
        "docker-compose")
            run_docker_compose "$2" "$3"
            ;;
        "run")
            run_flask_app
            ;;
        "train")
            run_training
            ;;
        "inference")
            run_inference "$2"
            ;;
        "test")
            run_tests
            ;;
        "clean")
            clean_environment
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@" 