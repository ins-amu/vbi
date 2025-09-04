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

# Function to ask for confirmation
confirm_action() {
    local message="$1"
    local default="${2:-n}"
    
    if [ "$default" = "y" ]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi
    
    echo -e "${YELLOW}[WARNING]${NC} $message"
    read -p "Do you want to continue? $prompt: " -n 1 -r
    echo    # Move to a new line
    
    if [ "$default" = "y" ]; then
        [[ $REPLY =~ ^[Nn]$ ]] && return 1 || return 0
    else
        [[ $REPLY =~ ^[Yy]$ ]] && return 0 || return 1
    fi
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

# Function to check if image exists (returns true/false without exiting)
image_exists() {
    docker image inspect $IMAGE_NAME > /dev/null 2>&1
}

# Function to build VBI Docker image
build_image() {
    local force_rebuild="$1"
    
    # Check if Dockerfile exists
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory: $(pwd)"
        print_info "Please run this script from the directory containing the Dockerfile."
        exit 1
    fi
    
    # Check if image already exists
    if image_exists && [ "$force_rebuild" != "--force" ] && [ "$force_rebuild" != "-f" ]; then
        print_warning "VBI Docker image '$IMAGE_NAME' already exists."
        echo ""
        if ! confirm_action "Do you want to rebuild it?"; then
            print_info "Build cancelled by user."
            exit 0
        fi
    fi
    
    print_info "Building VBI Docker image..."
    print_info "This may take several minutes..."
    
    # Build the image
    if docker build -t $IMAGE_NAME .; then
        print_success "VBI Docker image '$IMAGE_NAME' built successfully!"
        
        # Show image info
        print_info "Image details:"
        docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"
    else
        print_error "Failed to build VBI Docker image."
        exit 1
    fi
}

# Function to auto-build image if missing
auto_build_if_missing() {
    if ! image_exists; then
        print_warning "VBI Docker image '$IMAGE_NAME' not found."
        print_info "Attempting to build the image automatically..."
        echo ""
        
        # Check if Dockerfile exists
        if [ ! -f "Dockerfile" ]; then
            print_error "Dockerfile not found in current directory: $(pwd)"
            print_info "Please either:"
            print_info "  1. Run this script from the directory containing the Dockerfile"
            print_info "  2. Build the image manually: docker build -t $IMAGE_NAME ."
            exit 1
        fi
        
        build_image "--auto"
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
    
    # Auto-build image if missing
    auto_build_if_missing
    
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
    
    # Auto-build image if missing
    auto_build_if_missing
    
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
    
    # Auto-build image if missing
    auto_build_if_missing
    
    if command -v nvidia-smi > /dev/null 2>&1 && docker run --rm --gpus all hello-world > /dev/null 2>&1; then
        gpu_flag="--gpus all"
    fi
    
    print_info "Opening interactive shell in VBI container..."
    docker run --rm -it $gpu_flag $IMAGE_NAME /bin/bash
}

# Function to run a quick test
run_test() {
    local gpu_flag=""
    
    # Auto-build image if missing
    auto_build_if_missing
    
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
print('VBI Docker Test Results')
print('=' * 50)
print('VBI version:', vbi.__version__)
print()

# Run VBI comprehensive test imports
try:
    vbi.test_imports()
    print('\\n🎉 VBI Docker image is working correctly!')
except Exception as e:
    print('❌ Error running vbi.test_imports():', e)
    print('\\n❌ VBI Docker image test failed!')
"
}

# Function to remove containers
remove_containers() {
    print_info "Checking for VBI containers (stopped or running)..."
    
    # First, try to remove the named container
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Removing named container '${CONTAINER_NAME}'..."
        docker rm $CONTAINER_NAME
        print_success "Container '${CONTAINER_NAME}' removed."
    else
        print_warning "Named container '${CONTAINER_NAME}' does not exist."
    fi
    
    # Also check for any other containers using the VBI image
    VBI_CONTAINERS=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}} {{.Names}}" || true)
    if [ -n "$VBI_CONTAINERS" ]; then
        print_info "Found additional containers using VBI image:"
        echo "$VBI_CONTAINERS"
        print_info "Removing all containers using VBI image..."
        docker ps -a --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}}" | xargs -r docker rm
        print_success "All VBI containers removed."
    fi
}

# Function to stop all VBI containers
stop_all_containers() {
    print_info "Checking for running VBI containers..."
    
    # First, try to stop the named container
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Stopping named container '${CONTAINER_NAME}'..."
        docker stop $CONTAINER_NAME
        print_success "Container '${CONTAINER_NAME}' stopped."
    else
        print_warning "Named container '${CONTAINER_NAME}' is not running."
    fi
    
    # Also check for any other running containers using the VBI image
    RUNNING_VBI=$(docker ps --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}} {{.Names}}" || true)
    if [ -n "$RUNNING_VBI" ]; then
        print_info "Found additional running containers using VBI image:"
        echo "$RUNNING_VBI"
        print_info "Stopping all running VBI containers..."
        docker ps --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}}" | xargs -r docker stop
        print_success "All running VBI containers stopped."
    fi
}

# Function to remove VBI images
remove_images() {
    print_info "Checking for VBI images..."
    
    # Check for the specific image first
    if docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
        print_info "Removing image '${IMAGE_NAME}'..."
        docker rmi $IMAGE_NAME
        print_success "Image '${IMAGE_NAME}' removed."
    else
        # Check for any images that start with "vbi"
        VBI_IMAGES=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^vbi:" || true)
        if [ -n "$VBI_IMAGES" ]; then
            print_info "Found VBI images: $VBI_IMAGES"
            for image in $VBI_IMAGES; do
                print_info "Removing image '$image'..."
                docker rmi $image
                print_success "Image '$image' removed."
            done
        else
            print_warning "No VBI images found."
        fi
    fi
}

# Function to perform full cleanup
clean_all() {
    local force_flag="$1"
    
    if [ "$force_flag" != "--force" ] && [ "$force_flag" != "-f" ]; then
        print_warning "⚠️  DESTRUCTIVE OPERATION WARNING ⚠️"
        print_warning "This will:"
        print_warning "  • Stop ALL running VBI containers"
        print_warning "  • Remove ALL VBI containers"
        print_warning "  • Remove ALL VBI Docker images"
        print_warning "This action cannot be undone!"
        echo ""
        
        if ! confirm_action "You are about to perform a FULL VBI cleanup"; then
            print_info "Operation cancelled by user."
            exit 0
        fi
    fi
    
    print_info "Performing full VBI cleanup (containers + images)..."
    stop_all_containers
    remove_containers
    remove_images
    print_success "Full VBI cleanup completed."
}

# Function to remove containers only
remove_only() {
    local force_flag="$1"
    
    if [ "$force_flag" != "--force" ] && [ "$force_flag" != "-f" ]; then
        print_warning "⚠️  DESTRUCTIVE OPERATION WARNING ⚠️"
        print_warning "This will:"
        print_warning "  • Stop ALL running VBI containers"
        print_warning "  • Remove ALL VBI containers"
        print_warning "Docker images will be kept."
        echo ""
        
        if ! confirm_action "You are about to remove all VBI containers"; then
            print_info "Operation cancelled by user."
            exit 0
        fi
    fi
    
    print_info "Stopping and removing VBI containers..."
    stop_all_containers
    remove_containers
    print_success "VBI containers cleanup completed."
}

# Function to show image and container information
image_only() {
    print_info "VBI Docker Images and Containers Information"
    echo ""
    
    # Show VBI images
    print_info "📦 VBI Docker Images:"
    VBI_IMAGES=$(docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}" | grep "^vbi" || true)
    if [ -n "$VBI_IMAGES" ]; then
        echo "REPOSITORY          TAG       IMAGE ID       CREATED        SIZE"
        echo "$VBI_IMAGES"
    else
        print_warning "No VBI images found."
    fi
    echo ""
    
    # Show VBI containers (running and stopped)
    print_info "🐳 VBI Docker Containers:"
    VBI_CONTAINERS=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.CreatedAt}}" || true)
    if [ -n "$VBI_CONTAINERS" ]; then
        echo "$VBI_CONTAINERS"
    else
        print_warning "No VBI containers found."
    fi
    echo ""
    
    # Show named container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "📋 Named Container '${CONTAINER_NAME}' Status:"
        docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.CreatedAt}}"
    else
        print_warning "Named container '${CONTAINER_NAME}' does not exist."
    fi
}

# Function to show usage
show_usage() {
    echo "VBI Docker Container Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build [--force]          Build VBI Docker image"
    echo "  start [port]             Start interactive container (default port: 8888)"
    echo "  jupyter [port] [data_dir] Start Jupyter server (default port: 8888)"
    echo "  stop                     Stop the running container"
    echo "  restart [port]           Restart the interactive container"
    echo "  shell                    Open interactive shell"
    echo "  test                     Run functionality test"
    echo "  status                   Show container status"
    echo "  logs                     Show container logs"
    echo ""
    echo "Cleanup Commands (with confirmation prompts):"
    echo "  clean [--force]          Full cleanup (stop + remove containers + remove images)"
    echo "  remove [--force]         Stop and remove containers only (keep images)"
    echo ""
    echo "Information Commands:"
    echo "  image                    Show VBI images and containers information"
    echo ""
    echo "Options:"
    echo "  --force, -f              Skip confirmation prompts for cleanup commands"
    echo ""
    echo "Examples:"
    echo "  $0 build                 # Build VBI Docker image"
    echo "  $0 build --force         # Force rebuild VBI Docker image"
    echo "  $0 start                 # Start interactive container on port 8888 (auto-builds if needed)"
    echo "  $0 start 8889            # Start interactive container on port 8889"
    echo "  $0 jupyter               # Start Jupyter server on port 8888"
    echo "  $0 jupyter 8888 ./data   # Start Jupyter with data directory mounted"
    echo "  $0 shell                 # Open interactive shell"
    echo "  $0 test                  # Test VBI installation"
    echo "  $0 clean                 # Full cleanup (with confirmation)"
    echo "  $0 clean --force         # Full cleanup (skip confirmation)"
    echo "  $0 remove                # Remove containers only (with confirmation)"
    echo "  $0 image                 # Show VBI images and containers info"
}

# Main script logic
main() {
    check_docker
    
    # Only check for image existence for commands that need it and don't auto-build
    case "${1:-}" in
        build|clean|remove|image|help|--help|-h)
            # These commands don't require the image to exist or handle it themselves
            ;;
        *)
            # Other commands will auto-build if needed, so no check required here
            ;;
    esac
    
    case "${1:-}" in
        build)
            build_image "$2"
            ;;
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
        clean)
            clean_all "$2"
            ;;
        remove)
            remove_only "$2"
            ;;
        image)
            image_only
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
