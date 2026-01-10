#!/bin/bash
# Helper function to safely login to ECR Public with race condition prevention
# Uses flock for mutual exclusion and auth checking to minimize contention
#
# This script prevents the "device or resource busy" error that occurs when
# multiple Buildkite jobs try to docker login simultaneously on the same agent.
#
# Usage:
#   source docker_login_ecr_public.sh && safe_docker_login_ecr_public

set -euo pipefail

# Configuration
LOCK_FILE="/tmp/docker_ecr_public_login.lock"
LOCK_TIMEOUT=120  # seconds to wait for lock
AUTH_CHECK_TIMEOUT=5  # seconds for auth check
ECR_REGISTRY="public.ecr.aws"

check_docker_auth() {
    # Check if already authenticated to the given registry
    # Returns 0 if authenticated, 1 if not
    local registry="$1"

    # Try a lightweight operation to verify authentication
    # Using a dry-run pull with timeout to prevent hanging
    if timeout "$AUTH_CHECK_TIMEOUT" docker pull --dry-run "$registry/library/alpine:latest" >/dev/null 2>&1; then
        return 0
    fi

    return 1
}

safe_docker_login_ecr_public() {
    local registry="$ECR_REGISTRY"

    # Fast path: check if already authenticated
    echo "[docker-login] Checking authentication for $registry..."
    if check_docker_auth "$registry"; then
        echo "[docker-login] Already authenticated to $registry, skipping login"
        return 0
    fi

    # Need to login - acquire lock
    echo "[docker-login] Not authenticated, acquiring lock..."

    # Create lock file if doesn't exist
    touch "$LOCK_FILE"

    # Open lock file descriptor
    exec 200>"$LOCK_FILE"

    # Try to acquire lock with timeout
    local start_time=$(date +%s)
    local lock_acquired=0

    while [[ $lock_acquired -eq 0 ]]; do
        if flock -n 200; then
            lock_acquired=1
            echo "[docker-login] Lock acquired (PID: $$)"
            # Write PID to lock file for debugging
            echo "$$" >&200
        else
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))

            if [[ $elapsed -ge $LOCK_TIMEOUT ]]; then
                echo "[docker-login] ERROR: Timeout waiting for lock after ${LOCK_TIMEOUT}s" >&2
                exec 200>&-  # Close file descriptor
                return 1
            fi

            # Read current lock holder PID if available
            local lock_holder=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
            echo "[docker-login] Waiting for lock (held by PID: $lock_holder, elapsed: ${elapsed}s)..."
            sleep 1
        fi
    done

    # Double-check authentication (another process may have logged in while we waited)
    if check_docker_auth "$registry"; then
        echo "[docker-login] Already authenticated (logged in by another process), releasing lock"
        flock -u 200
        exec 200>&-
        return 0
    fi

    # Actually perform login
    echo "[docker-login] Performing docker login to $ECR_REGISTRY..."
    if aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ECR_REGISTRY"; then
        echo "[docker-login] Login successful"
        flock -u 200
        exec 200>&-
        return 0
    else
        local exit_code=$?
        echo "[docker-login] ERROR: Login failed with exit code $exit_code" >&2
        flock -u 200
        exec 200>&-
        return $exit_code
    fi
}

# Execute if run as script (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    safe_docker_login_ecr_public
fi
