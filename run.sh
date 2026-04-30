#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AgriSense — Docker Build & Run Script
# ═══════════════════════════════════════════════════════════════
# Usage:
#   ./run.sh build     → Build Docker image
#   ./run.sh run       → Run container
#   ./run.sh dev       → Run in development mode (with live logs)
#   ./run.sh stop      → Stop running container
#   ./run.sh push      → Push to DockerHub
#   ./run.sh clean     → Remove image and container
# ═══════════════════════════════════════════════════════════════

set -e  # Exit immediately on error

# ── Configuration
DOCKERHUB_USERNAME="your-dockerhub-username"  # ← Change this
IMAGE_NAME="agrisense"
IMAGE_TAG="latest"
FULL_IMAGE="${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="agrisense-app"
PORT=5000

# ── Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

log_info()    { echo -e "${CYAN}[INFO]${RESET} $1"; }
log_success() { echo -e "${GREEN}[✅]${RESET} $1"; }
log_warn()    { echo -e "${YELLOW}[⚠️]${RESET} $1"; }
log_error()   { echo -e "${RED}[❌]${RESET} $1"; }

# ═══════════════════════════════════════════════════════════════
build() {
    log_info "Building AgriSense Docker image..."
    log_info "Image: ${FULL_IMAGE}"
    echo ""
    
    docker build \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
        --tag "${FULL_IMAGE}" \
        --progress=plain \
        .
    
    log_success "Build complete! Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    docker images "${IMAGE_NAME}"
}

# ═══════════════════════════════════════════════════════════════
run() {
    # Check if container already running
    if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
        log_warn "Container '${CONTAINER_NAME}' is already running!"
        log_info "Run './run.sh stop' first, then try again"
        exit 1
    fi

    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_warn ".env file not found. Creating from .env.example..."
        cp .env.example .env
        log_warn "Please edit .env and add your GROQ_API_KEY before using the chatbot!"
    fi

    log_info "Starting AgriSense container..."
    
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --port "${PORT}:5000" \
        --env-file .env \
        --volume "$(pwd)/uploads:/app/uploads" \
        --restart unless-stopped \
        "${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_success "Container started!"
    log_info "URL: http://localhost:${PORT}"
    log_info "Health: http://localhost:${PORT}/health"
    log_info "Logs: docker logs -f ${CONTAINER_NAME}"
}

# ═══════════════════════════════════════════════════════════════
dev() {
    log_info "Running AgriSense in development mode (attached)..."
    
    if [ ! -f ".env" ]; then
        cp .env.example .env
    fi
    
    docker run --rm \
        --name "${CONTAINER_NAME}-dev" \
        -p "${PORT}:5000" \
        --env-file .env \
        -e FLASK_DEBUG=true \
        -v "$(pwd)/app.py:/app/app.py" \
        -v "$(pwd)/static:/app/static" \
        -v "$(pwd)/templates:/app/templates" \
        -v "$(pwd)/uploads:/app/uploads" \
        "${IMAGE_NAME}:${IMAGE_TAG}"
}

# ═══════════════════════════════════════════════════════════════
stop() {
    log_info "Stopping AgriSense container..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null && \
    docker rm "${CONTAINER_NAME}" 2>/dev/null && \
    log_success "Container stopped and removed!" || \
    log_warn "Container was not running"
}

# ═══════════════════════════════════════════════════════════════
push() {
    log_info "Pushing to DockerHub as: ${FULL_IMAGE}"
    log_warn "Make sure you're logged in: docker login"
    
    docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"
    
    log_success "Pushed to DockerHub!"
    log_info "Pull with: docker pull ${FULL_IMAGE}"
}

# ═══════════════════════════════════════════════════════════════
clean() {
    log_warn "This will remove the container and image. Are you sure? (y/n)"
    read -r confirm
    if [ "$confirm" = "y" ]; then
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        docker rm "${CONTAINER_NAME}" 2>/dev/null || true
        docker rmi "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null || true
        log_success "Cleaned up!"
    fi
}

# ═══════════════════════════════════════════════════════════════
logs() {
    docker logs -f "${CONTAINER_NAME}"
}

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
echo ""
echo "  🌾 AgriSense — Docker Manager"
echo "  ─────────────────────────────"

case "${1:-help}" in
    build)  build  ;;
    run)    run    ;;
    dev)    dev    ;;
    stop)   stop   ;;
    push)   push   ;;
    clean)  clean  ;;
    logs)   logs   ;;
    *)
        echo "  Usage: ./run.sh [command]"
        echo ""
        echo "  Commands:"
        echo "    build  → Build Docker image"
        echo "    run    → Run container in background"
        echo "    dev    → Run container with live reload"
        echo "    stop   → Stop and remove container"
        echo "    push   → Push image to DockerHub"
        echo "    logs   → Follow container logs"
        echo "    clean  → Remove image and container"
        echo ""
        ;;
esac
