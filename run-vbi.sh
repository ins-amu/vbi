#!/bin/bash

# VBI Docker Container Startup Script
# This script provides easy ways to run the VBI Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="vbi:latest"
CONTAINER_NAME="vbi-workspace"
DEFAULT_PORT=8888

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if image exists
check_image() {
    if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
        print_error "VBI Docker image '$IMAGE_NAME' not found."
        print_info "Please build the image first with: docker build -t $IMAGE_NAME ."
        exit 1
    fi
}

# Function to check if container is already running
check_running() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        return 0
    else
        return 1
    fi
}

# Function to get container status
get_container_logs() {
    if check_running; then
        print_info "Container logs:"
        docker logs $CONTAINER_NAME --tail 10
    fi
}

# Function to start interactive container
start_container() {
    local port=${1:-$DEFAULT_PORT}
    local gpu_flag=""
    
    # Check for GPU support
    if command -v nvidia-smi > /dev/null 2>&1 && docker run --rm --gpus all hello-world > /dev/null 2>&1; then
        gpu_flag="--gpus all"
        print_info "GPU support detected and enabled"
    else
        print_warning "GPU support not available, running in CPU mode"
    fi
    
    # Build docker run command
    local docker_cmd="docker run -it --rm"
    docker_cmd="$docker_cmd $gpu_flag -p $port:8888"
    
    # Mount current directory for workspace
    docker_cmd="$docker_cmd -v $(pwd):/app/workspace"
    
    docker_cmd="$docker_cmd $IMAGE_NAME"
    
    print_info "Starting VBI interactive container..."
    print_info "Port $port is mapped to container port 8888"
    print_info "Current directory mounted at: /app/workspace"
    print_info "Command: $docker_cmd"
    print_info ""
    
    # Execute the command
    eval $docker_cmd
}

# Function to start Jupyter server
start_jupyter() {
    local port=${1:-$DEFAULT_PORT}
    local data_dir=${2:-""}
    local gpu_flag=""
    
    # Check for GPU support
    if command -v nvidia-smi > /dev/null 2>&1 && docker run --rm --gpus all hello-world > /dev/null 2>&1; then
        gpu_flag="--gpus all"
        print_info "GPU support detected and enabled"
    else
        print_warning "GPU support not available, running in CPU mode"
    fi
    
    # Build docker run command
    local docker_cmd="docker run --rm -d --name $CONTAINER_NAME"
    docker_cmd="$docker_cmd $gpu_flag -p $port:8888"
    
    # Add volume mounts if data directory specified
    if [ ! -z "$data_dir" ]; then
        if [ ! -d "$data_dir" ]; then
            print_error "Data directory '$data_dir' does not exist"
            exit 1
        fi
        docker_cmd="$docker_cmd -v $(realpath $data_dir):/app/data"
        print_info "Mounting data directory: $data_dir -> /app/data"
    fi
    
    # Mount current directory for notebooks
    docker_cmd="$docker_cmd -v $(pwd):/app/workspace"
    
    docker_cmd="$docker_cmd $IMAGE_NAME jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/app"
    
    print_info "Starting VBI Jupyter server..."
    eval $docker_cmd
    
    # Wait for Jupyter to start
    print_info "Waiting for Jupyter to start..."
    sleep 5
    
    # Get the token from logs
    local logs=$(docker logs $CONTAINER_NAME 2>&1)
    local token=$(echo "$logs" | grep -oP '(?<=token=)[a-f0-9]+' | head -1)
    
    print_success "VBI Jupyter server started successfully!"
    print_info "Container name: $CONTAINER_NAME"
    print_info "Local port: $port"
    print_info "Access URL: http://localhost:$port"
    if [ ! -z "$token" ]; then
        print_info "Full URL: http://localhost:$port/lab?token=$token"
    fi
    print_info ""
    print_info "Your current directory is mounted at: /app/workspace"
    if [ ! -z "$data_dir" ]; then
        print_info "Your data directory is mounted at: /app/data"
    fi
    print_info ""
    print_info "To stop the server: $0 stop"
    print_info "To view logs: $0 logs"
}

# Function to stop container
stop_container() {
    if check_running; then
        print_info "Stopping VBI container..."
        docker stop $CONTAINER_NAME > /dev/null
        print_success "VBI container stopped"
    else
        print_warning "VBI container is not running"
    fi
}

# Function to open shell
open_shell() {
    local gpu_flag=""
    
    if command -v nvidia-smi > /dev/null 2>&1 && docker run --rm --gpus all hello-world > /dev/null 2>&1; then
        gpu_flag="--gpus all"
    fi
    
    print_info "Opening interactive shell in VBI container..."
    docker run --rm -it $gpu_flag $IMAGE_NAME /bin/bash
}

# Function to run a quick test
run_test() {
    local gpu_flag=""
    
    # Check for GPU support
    if command -v nvidia-smi > /dev/null 2>&1 && docker run --rm --gpus all hello-world > /dev/null 2>&1; then
        gpu_flag="--gpus all"
        print_info "GPU support detected, testing with GPU access"
    else
        print_warning "GPU support not available, testing in CPU mode"
    fi
    
    print_info "Running VBI functionality test..."
    docker run --rm $gpu_flag $IMAGE_NAME python -c "
import vbi
print('✅ VBI version:', vbi.__version__)

try:
    import torch
    print('✅ PyTorch imported successfully')
    print('   CUDA available:', torch.cuda.is_available())
except Exception as e:
    print('❌ PyTorch error:', e)

try:
    import cupy
    print('✅ CuPy imported successfully')
    print('   CuPy version:', cupy.__version__)
    # Test basic CuPy functionality
    import cupy as cp
    x = cp.array([1, 2, 3])
    print('   CuPy basic test passed')
except Exception as e:
    print('❌ CuPy error:', e)

try:
    import numpy as np
    import scipy
    import matplotlib
    print('✅ Scientific packages (NumPy, SciPy, Matplotlib) imported successfully')
except Exception as e:
    print('❌ Scientific packages error:', e)

print('\\n🎉 VBI Docker image is working correctly!')
"
}

# Function to show usage
show_usage() {
    echo "VBI Docker Container Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [port]             Start interactive container (default port: 8888)"
    echo "  jupyter [port] [data_dir] Start Jupyter server (default port: 8888)"
    echo "  stop                     Stop the running container"
    echo "  restart [port]           Restart the interactive container"
    echo "  shell                    Open interactive shell"
    echo "  test                     Run functionality test"
    echo "  status                   Show container status"
    echo "  logs                     Show container logs"
    echo ""
    echo "Examples:"
    echo "  $0 start                 # Start interactive container on port 8888"
    echo "  $0 start 8889            # Start interactive container on port 8889"
    echo "  $0 jupyter               # Start Jupyter server on port 8888"
    echo "  $0 jupyter 8888 ./data   # Start Jupyter with data directory mounted"
    echo "  $0 shell                 # Open interactive shell"
    echo "  $0 test                  # Test VBI installation"
}

# Main script logic
main() {
    check_docker
    check_image
    
    case "${1:-}" in
        start)
            start_container "$2"
            ;;
        jupyter)
            if check_running; then
                print_warning "Container '$CONTAINER_NAME' is already running"
                get_container_logs
                exit 0
            fi
            start_jupyter "$2" "$3"
            ;;
        stop)
            stop_container
            ;;
        restart)
            stop_container
            sleep 2
            start_container "$2"
            ;;
        shell)
            open_shell
            ;;
        test)
            run_test
            ;;
        status)
            if check_running; then
                print_success "Container '$CONTAINER_NAME' is running"
                docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
            else
                print_info "Container '$CONTAINER_NAME' is not running"
            fi
            ;;
        logs)
            get_container_logs
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: ${1:-}"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
