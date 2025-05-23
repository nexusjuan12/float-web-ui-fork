#!/bin/bash

# FLOAT Web UI Deployment Script for Ubuntu 22.04 + CUDA 12.1
# Designed for vast.ai template deployment

# Ensure we're running with bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash. Please run with: bash $0"
    exit 1
fi

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Configuration
CONDA_DIR="$HOME/miniconda3"
CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
REPO_URL="https://github.com/nexusjuan12/float-web-ui-fork.git"
PROJECT_DIR="$HOME/FLOAT-web-ui"
ENV_NAME="float-web-ui"
PYTHON_VERSION="3.9"  # Changed from 3.8.5 to 3.9 for better compatibility

# Check system requirements
check_system() {
    log "Checking system requirements..."
    
    # Check CUDA version
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        log "Detected CUDA Version: $CUDA_VERSION"
        
        if [[ ! $CUDA_VERSION == 12.* ]]; then
            warn "Expected CUDA 12.x, found $CUDA_VERSION. Proceeding anyway..."
        fi
    else
        error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
        exit 1
    fi
    
    # Check available disk space (need at least 10GB)
    AVAILABLE_SPACE=$(df -BG "$HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 10 ]; then
        error "Insufficient disk space. Need at least 10GB, have ${AVAILABLE_SPACE}GB"
        exit 1
    fi
    
    # Install system dependencies
    log "Installing system dependencies..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y ffmpeg git wget curl
    elif command -v yum &> /dev/null; then
        sudo yum install -y ffmpeg git wget curl
    else
        warn "Package manager not found. Please install ffmpeg, git, wget, and curl manually."
    fi
    
    # Verify ffmpeg installation
    if command -v ffmpeg &> /dev/null && command -v ffprobe &> /dev/null; then
        log "ffmpeg and ffprobe installed successfully ✓"
    else
        warn "ffmpeg/ffprobe installation may have failed. Audio processing might not work."
    fi
    
    log "System check passed ✓"
}

# Install conda if not present
install_conda() {
    if [ -d "$CONDA_DIR" ] && [ -f "$CONDA_DIR/bin/conda" ]; then
        log "Conda already installed at $CONDA_DIR"
        return 0
    fi
    
    log "Installing Miniconda..."
    
    # Download Miniconda installer
    TEMP_INSTALLER=$(mktemp --suffix=.sh)
    wget -q --show-progress "$CONDA_URL" -O "$TEMP_INSTALLER"
    
    # Explicitly run with bash to avoid the sourcing issue
    /bin/bash "$TEMP_INSTALLER" -b -p "$CONDA_DIR" -u
    
    # Clean up installer
    rm "$TEMP_INSTALLER"
    
    # Add conda to PATH for current session
    export PATH="$CONDA_DIR/bin:$PATH"
    
    # Initialize conda for current shell
    eval "$("$CONDA_DIR/bin/conda" shell.bash hook)"
    
    # Initialize conda for future shells (suppress output)
    "$CONDA_DIR/bin/conda" init bash >/dev/null 2>&1 || true
    
    log "Conda installation completed ✓"
}

# Setup conda environment
setup_environment() {
    log "Setting up conda environment..."
    
    # Ensure conda is in PATH
    export PATH="$CONDA_DIR/bin:$PATH"
    
    # Initialize conda for current shell
    eval "$($CONDA_DIR/bin/conda shell.bash hook)"
    
    # Create environment if it doesn't exist
    if conda env list | grep -q "^$ENV_NAME "; then
        log "Environment $ENV_NAME already exists"
    else
        log "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    fi
    
    # Activate environment
    conda activate "$ENV_NAME"
    
    log "Environment setup completed ✓"
}

# Clone repository
clone_repository() {
    log "Cloning FLOAT repository..."
    
    if [ -d "$PROJECT_DIR" ]; then
        warn "Directory $PROJECT_DIR already exists. Pulling latest changes..."
        cd "$PROJECT_DIR"
        git pull
    else
        git clone "$REPO_URL" "$PROJECT_DIR"
        cd "$PROJECT_DIR"
    fi
    
    log "Repository cloned ✓"
}

# Install PyTorch with CUDA 12.1 support
install_pytorch() {
    log "Installing PyTorch with CUDA 12.1 support..."
    
    # Ensure conda is in PATH and activated
    export PATH="$CONDA_DIR/bin:$PATH"
    eval "$($CONDA_DIR/bin/conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    # Install PyTorch with CUDA 12.1 support
    # Using the latest stable PyTorch that supports CUDA 12.1
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    
    # Verify installation
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    
    log "PyTorch installation completed ✓"
}

# Install requirements
install_requirements() {
    log "Installing project requirements..."
    
    # Ensure conda is in PATH and activated
    export PATH="$CONDA_DIR/bin:$PATH"
    eval "$($CONDA_DIR/bin/conda shell.bash hook)"
    conda activate "$ENV_NAME"
    
    cd "$PROJECT_DIR"
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        warn "requirements.txt not found. Installing common dependencies..."
        # Install common ML/CV dependencies
        pip install \
            numpy \
            opencv-python \
            pillow \
            scipy \
            scikit-image \
            matplotlib \
            tqdm \
            gradio \
            transformers \
            diffusers \
            accelerate \
            xformers
    fi
    
    log "Requirements installation completed ✓"
}

# Download checkpoints
download_checkpoints() {
    log "Setting up checkpoints..."
    
    cd "$PROJECT_DIR"
    
    # Create checkpoints directory
    mkdir -p checkpoints
    
    # Check if download script exists and run it
    if [ -f "download_checkpoints.sh" ]; then
        log "Running checkpoint download script..."
        chmod +x download_checkpoints.sh
        ./download_checkpoints.sh
    else
        warn "download_checkpoints.sh not found. Please download checkpoints manually."
        echo "Manual download instructions:"
        echo "1. Download from: https://drive.google.com/file/d/1rvWuM12cyvNvBQNCLmG4Fr2L1rpjQBF0/view?usp=sharing"
        echo "2. Extract to: $PROJECT_DIR/checkpoints/"
    fi
    
    log "Checkpoint setup completed ✓"
}

# Create startup script
create_startup_script() {
    log "Creating startup script..."
    
    cat > "$PROJECT_DIR/start_float.sh" << 'EOF'
#!/bin/bash

# FLOAT Web UI Startup Script

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting FLOAT Web UI...${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
source "$HOME/miniconda3/bin/activate"
conda activate float-web-ui

# Check if checkpoints exist
if [ ! -f "checkpoints/float.pth" ]; then
    echo -e "${GREEN}Checkpoints not found. Please download them manually.${NC}"
    echo "See README for instructions."
    exit 1
fi

# Start the web UI
echo -e "${GREEN}Launching FLOAT Web UI on port 7860...${NC}"
echo -e "${GREEN}Access at: http://localhost:7860${NC}"
echo -e "${GREEN}For public access, use: --share flag${NC}"

python app.py --port 7860 --share

EOF
    
    chmod +x "$PROJECT_DIR/start_float.sh"
    
    log "Startup script created at $PROJECT_DIR/start_float.sh ✓"
}

# Create environment activation script
create_env_script() {
    log "Creating environment activation script..."
    
    cat > "$PROJECT_DIR/activate_env.sh" << 'EOF'
#!/bin/bash

# Activate FLOAT environment
source "$HOME/miniconda3/bin/activate"
conda activate float-web-ui

echo "FLOAT environment activated!"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Launch bash in the activated environment
exec bash

EOF
    
    chmod +x "$PROJECT_DIR/activate_env.sh"
    
    log "Environment activation script created ✓"
}

# Main deployment function
main() {
    log "Starting FLOAT Web UI deployment..."
    log "Target directory: $PROJECT_DIR"
    log "Environment name: $ENV_NAME"
    
    check_system
    install_conda
    setup_environment
    clone_repository
    install_pytorch
    install_requirements
    download_checkpoints
    create_startup_script
    create_env_script
    
    log "🎉 Deployment completed successfully!"
    echo
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Start the web UI: cd $PROJECT_DIR && ./start_float.sh"
    echo "2. Or activate environment: cd $PROJECT_DIR && ./activate_env.sh"
    echo "3. Access web UI at: http://localhost:7860"
    echo
    echo -e "${BLUE}Usage:${NC}"
    echo "- Upload a portrait image (512x512 recommended)"
    echo "- Upload audio file (WAV, 16kHz recommended)"
    echo "- Select emotion and intensity"
    echo "- Click Generate"
    echo
    echo -e "${YELLOW}Note: Make sure to download checkpoints if the automatic download failed${NC}"
}

# Run main function
main "$@"