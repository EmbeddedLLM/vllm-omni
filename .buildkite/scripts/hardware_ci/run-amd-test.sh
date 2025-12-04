#!/bin/bash

# This script runs tests inside the ROCm docker container for vLLM-Omni.
# Adapted from vLLM's run-amd-test.sh for vllm-omni's simpler use case.

set -o pipefail

# Export Python path
export PYTHONPATH=".."

# Print ROCm version
echo "--- Confirming Clean Initial State"
while true; do
    sleep 3
    if grep -q clean /opt/amdgpu/etc/gpu_state; then
        echo "GPUs state is \"clean\""
        break
    fi
done

echo "--- ROCm info"
rocminfo

# Cleanup older docker images
cleanup_docker() {
    docker_root=$(docker info -f '{{.DockerRootDir}}')
    if [ -z "$docker_root" ]; then
        echo "Failed to determine Docker root directory."
        exit 1
    fi
    echo "Docker root directory: $docker_root"

    disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
    threshold=70

    if [ "$disk_usage" -gt "$threshold" ]; then
        echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
        docker image prune -f
        docker volume prune -f && docker system prune --force --filter "until=72h" --all
        echo "Docker images and volumes cleanup completed."
    else
        echo "Disk usage is below $threshold%. No cleanup needed."
    fi
}

cleanup_docker

echo "--- Resetting GPUs"
echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
    sleep 3
    if grep -q clean /opt/amdgpu/etc/gpu_state; then
        echo "GPUs state is \"clean\""
        break
    fi
done

echo "--- Pulling/Building container"
image_name="rocm/vllm-omni-ci:${BUILDKITE_COMMIT}"
container_name="rocm_vllm_omni_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# Try to pull image first, if it doesn't exist, build it
if ! docker pull "${image_name}" 2>/dev/null; then
    echo "Image not found, building from Dockerfile.rocm..."
    cd "$(dirname "$0")/../../.."  # Go to repo root
    docker build \
        -f docker/Dockerfile.rocm \
        -t "${image_name}" \
        --build-arg BUILDKITE_COMMIT="${BUILDKITE_COMMIT}" \
        .
fi

remove_docker_container() {
    docker rm -f "${container_name}" || docker image rm -f "${image_name}" || true
}
trap remove_docker_container EXIT

echo "--- Running container"

# HuggingFace cache setup
HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

# Get commands from arguments
commands=$@
echo "Commands: $commands"

# Get render group for GPU access
render_gid=$(getent group render | cut -d: -f3)
if [[ -z "$render_gid" ]]; then
    echo "Error: 'render' group not found. This is required for GPU access." >&2
    exit 1
fi

# Run tests in container
echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"
docker run \
    --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
    --network=host \
    --shm-size=16gb \
    --group-add "$render_gid" \
    --rm \
    -e HF_TOKEN \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -v "${HF_CACHE}:${HF_MOUNT}" \
    -e "HF_HOME=${HF_MOUNT}" \
    -e "PYTHONPATH=.." \
    --name "${container_name}" \
    "${image_name}" \
    /bin/bash -c "${commands}"
