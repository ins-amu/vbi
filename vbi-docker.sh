#!/bin/bash

# VBI Docker Image Builder and Container Manager

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
    echo

    if [ "$default" = "y" ]; then
        [[ $REPLY =~ ^[Nn]$ ]] && return 1 || return 0
    else
        [[ $REPLY =~ ^[Yy]$ ]] && return 0 || return 1
    fi
}

check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Returns 0 if host has nvidia-smi and Docker has the nvidia runtime registered
check_gpu_support() {
    command -v nvidia-smi > /dev/null 2>&1 && \
        docker info 2>/dev/null | grep -qi "nvidia"
}

check_image() {
    if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
        print_error "VBI Docker image '$IMAGE_NAME' not found."
        print_info "Please build the image first with: docker build -t $IMAGE_NAME ."
        exit 1
    fi
}

image_exists() {
    docker image inspect "$IMAGE_NAME" > /dev/null 2>&1
}

build_image() {
    local force_rebuild="$1"

    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found in current directory: $(pwd)"
        print_info "Please run this script from the directory containing the Dockerfile."
        exit 1
    fi

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

    if docker build -t "$IMAGE_NAME" .; then
        print_success "VBI Docker image '$IMAGE_NAME' built successfully!"
        print_info "Image details:"
        docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"
    else
        print_error "Failed to build VBI Docker image."
        exit 1
    fi
}

auto_build_if_missing() {
    if ! image_exists; then
        print_warning "VBI Docker image '$IMAGE_NAME' not found."
        print_info "Attempting to build the image automatically..."
        echo ""

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

check_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

get_container_logs() {
    if check_running; then
        print_info "Container logs:"
        docker logs "$CONTAINER_NAME" --tail 10
    fi
}

start_container() {
    local port=${1:-$DEFAULT_PORT}

    auto_build_if_missing

    local -a docker_cmd=(docker run -it --rm)

    if check_gpu_support; then
        docker_cmd+=("--gpus" "all")
        print_info "GPU support detected and enabled"
    else
        print_warning "GPU support not available, running in CPU mode"
    fi

    docker_cmd+=("-p" "$port:8888" "-v" "$(pwd):/app/workspace" "$IMAGE_NAME")

    print_info "Starting VBI interactive container..."
    print_info "Port $port is mapped to container port 8888"
    print_info "Current directory mounted at: /app/workspace"
    print_info ""

    "${docker_cmd[@]}"
}

start_jupyter() {
    local port=${1:-$DEFAULT_PORT}
    local data_dir=${2:-""}

    auto_build_if_missing

    local -a docker_cmd=(docker run --rm -d --name "$CONTAINER_NAME")

    if check_gpu_support; then
        docker_cmd+=("--gpus" "all")
        print_info "GPU support detected and enabled"
    else
        print_warning "GPU support not available, running in CPU mode"
    fi

    docker_cmd+=("-p" "$port:8888" "-v" "$(pwd):/app/workspace")

    if [ -n "$data_dir" ]; then
        if [ ! -d "$data_dir" ]; then
            print_error "Data directory '$data_dir' does not exist"
            exit 1
        fi
        docker_cmd+=("-v" "$(realpath "$data_dir"):/app/data")
        print_info "Mounting data directory: $data_dir -> /app/data"
    fi

    docker_cmd+=("$IMAGE_NAME" "jupyter" "lab" "--ip=0.0.0.0" "--port=8888" "--no-browser" "--allow-root" "--notebook-dir=/app")

    print_info "Starting VBI Jupyter server..."
    "${docker_cmd[@]}"

    print_info "Waiting for Jupyter to start..."
    sleep 5

    local logs token
    logs=$(docker logs "$CONTAINER_NAME" 2>&1)
    token=$(echo "$logs" | grep -oP '(?<=token=)[a-f0-9]+' | head -1)

    print_success "VBI Jupyter server started successfully!"
    print_info "Container name: $CONTAINER_NAME"
    print_info "Local port: $port"
    print_info "Access URL: http://localhost:$port"
    if [ -n "$token" ]; then
        print_info "Full URL: http://localhost:$port/lab?token=$token"
    fi
    print_info ""
    print_info "Your current directory is mounted at: /app/workspace"
    if [ -n "$data_dir" ]; then
        print_info "Your data directory is mounted at: /app/data"
    fi
    print_info ""
    print_info "To stop the server: $0 stop"
    print_info "To view logs: $0 logs"
}

stop_container() {
    if check_running; then
        print_info "Stopping VBI container..."
        docker stop "$CONTAINER_NAME" > /dev/null
        print_success "VBI container stopped"
    else
        print_warning "VBI container is not running"
    fi
}

open_shell() {
    auto_build_if_missing

    local -a docker_cmd=(docker run --rm -it)

    if check_gpu_support; then
        docker_cmd+=("--gpus" "all")
    fi

    print_info "Opening interactive shell in VBI container..."
    "${docker_cmd[@]}" "$IMAGE_NAME" /bin/bash
}

run_test() {
    auto_build_if_missing

    local -a docker_cmd=(docker run --rm)

    if check_gpu_support; then
        docker_cmd+=("--gpus" "all")
        print_info "GPU support detected, testing with GPU access"
    else
        print_warning "GPU support not available, testing in CPU mode"
    fi

    print_info "Running VBI functionality test..."
    "${docker_cmd[@]}" "$IMAGE_NAME" python -c "
import vbi
print('VBI Docker Test Results')
print('=' * 50)
print('VBI version:', vbi.__version__)
print()

try:
    vbi.test_imports()
    print('\n VBI Docker image is working correctly!')
except Exception as e:
    print(' Error running vbi.test_imports():', e)
    print('\n VBI Docker image test failed!')
"
}

remove_containers() {
    print_info "Checking for VBI containers (stopped or running)..."

    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Removing named container '${CONTAINER_NAME}'..."
        docker rm "$CONTAINER_NAME"
        print_success "Container '${CONTAINER_NAME}' removed."
    else
        print_warning "Named container '${CONTAINER_NAME}' does not exist."
    fi

    local vbi_containers
    vbi_containers=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}} {{.Names}}" || true)
    if [ -n "$vbi_containers" ]; then
        print_info "Found additional containers using VBI image:"
        echo "$vbi_containers"
        print_info "Removing all containers using VBI image..."
        docker ps -a --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}}" | xargs -r docker rm
        print_success "All VBI containers removed."
    fi
}

stop_all_containers() {
    print_info "Checking for running VBI containers..."

    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Stopping named container '${CONTAINER_NAME}'..."
        docker stop "$CONTAINER_NAME"
        print_success "Container '${CONTAINER_NAME}' stopped."
    else
        print_warning "Named container '${CONTAINER_NAME}' is not running."
    fi

    local running_vbi
    running_vbi=$(docker ps --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}} {{.Names}}" || true)
    if [ -n "$running_vbi" ]; then
        print_info "Found additional running containers using VBI image:"
        echo "$running_vbi"
        print_info "Stopping all running VBI containers..."
        docker ps --filter "ancestor=${IMAGE_NAME}" --format "{{.ID}}" | xargs -r docker stop
        print_success "All running VBI containers stopped."
    fi
}

remove_images() {
    print_info "Checking for VBI images..."

    if docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
        print_info "Removing image '${IMAGE_NAME}'..."
        docker rmi "$IMAGE_NAME"
        print_success "Image '${IMAGE_NAME}' removed."
    else
        local vbi_images
        vbi_images=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^vbi:" || true)
        if [ -n "$vbi_images" ]; then
            print_info "Found VBI images: $vbi_images"
            for image in $vbi_images; do
                print_info "Removing image '$image'..."
                docker rmi "$image"
                print_success "Image '$image' removed."
            done
        else
            print_warning "No VBI images found."
        fi
    fi
}

clean_all() {
    local force_flag="$1"

    if [ "$force_flag" != "--force" ] && [ "$force_flag" != "-f" ]; then
        print_warning "DESTRUCTIVE OPERATION WARNING"
        print_warning "This will:"
        print_warning "  - Stop ALL running VBI containers"
        print_warning "  - Remove ALL VBI containers"
        print_warning "  - Remove ALL VBI Docker images"
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

remove_only() {
    local force_flag="$1"

    if [ "$force_flag" != "--force" ] && [ "$force_flag" != "-f" ]; then
        print_warning "DESTRUCTIVE OPERATION WARNING"
        print_warning "This will:"
        print_warning "  - Stop ALL running VBI containers"
        print_warning "  - Remove ALL VBI containers"
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

image_only() {
    print_info "VBI Docker Images and Containers Information"
    echo ""

    print_info "VBI Docker Images:"
    local vbi_images
    vbi_images=$(docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}" | grep "^vbi" || true)
    if [ -n "$vbi_images" ]; then
        echo "REPOSITORY          TAG       IMAGE ID       CREATED        SIZE"
        echo "$vbi_images"
    else
        print_warning "No VBI images found."
    fi
    echo ""

    print_info "VBI Docker Containers:"
    local vbi_containers
    vbi_containers=$(docker ps -a --filter "ancestor=${IMAGE_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.CreatedAt}}" || true)
    if [ -n "$vbi_containers" ]; then
        echo "$vbi_containers"
    else
        print_warning "No VBI containers found."
    fi
    echo ""

    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Named Container '${CONTAINER_NAME}' Status:"
        docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.CreatedAt}}"
    else
        print_warning "Named container '${CONTAINER_NAME}' does not exist."
    fi
}

show_usage() {
    echo "VBI Docker Image Builder and Container Manager"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  build [--force]           Build VBI Docker image"
    echo "  start [port]              Start interactive container (default port: 8888)"
    echo "  jupyter [port] [data_dir] Start Jupyter server (default port: 8888)"
    echo "  stop                      Stop the running container"
    echo "  restart [port]            Restart the interactive container"
    echo "  shell                     Open interactive shell"
    echo "  test                      Run functionality test"
    echo "  status                    Show container status"
    echo "  logs                      Show container logs"
    echo ""
    echo "Cleanup Commands (with confirmation prompts):"
    echo "  clean [--force]           Full cleanup (stop + remove containers + remove images)"
    echo "  remove [--force]          Stop and remove containers only (keep images)"
    echo ""
    echo "Information Commands:"
    echo "  image                     Show VBI images and containers information"
    echo ""
    echo "Options:"
    echo "  --force, -f               Skip confirmation prompts for cleanup commands"
    echo ""
    echo "Examples:"
    echo "  $0 build                  # Build VBI Docker image"
    echo "  $0 build --force          # Force rebuild VBI Docker image"
    echo "  $0 start                  # Start interactive container on port 8888"
    echo "  $0 start 8889             # Start interactive container on port 8889"
    echo "  $0 jupyter                # Start Jupyter server on port 8888"
    echo "  $0 jupyter 8888 ./data    # Start Jupyter with data directory mounted"
    echo "  $0 shell                  # Open interactive shell"
    echo "  $0 test                   # Test VBI installation"
    echo "  $0 clean                  # Full cleanup (with confirmation)"
    echo "  $0 clean --force          # Full cleanup (skip confirmation)"
    echo "  $0 remove                 # Remove containers only (with confirmation)"
    echo "  $0 image                  # Show VBI images and containers info"
    echo ""
    echo "Help:"
    echo "  $0 help | --help | -h     Show this help message"
}

main() {
    check_docker

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
        "")
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
