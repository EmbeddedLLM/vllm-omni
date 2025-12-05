#!/bin/bash
# Simplified AMD CI bootstrap for vllm-omni
# Based on vllm-project/ci-infra bootstrap-amd.sh but adapted for vllm-omni
#
# This script:
# 1. Downloads the AMD pipeline template from ci-infra
# 2. Processes test-amd.yaml using minijinja
# 3. Uploads the generated pipeline to Buildkite

set -euo pipefail

# Configuration with defaults
VLLM_CI_BRANCH="${VLLM_CI_BRANCH:-main}"
AMD_MIRROR_HW="${AMD_MIRROR_HW:-amdexperimental}"
RUN_ALL="${RUN_ALL:-1}"  # Default to run all tests for vllm-omni
NIGHTLY="${NIGHTLY:-0}"
COV_ENABLED="${COV_ENABLED:-0}"

echo "--- AMD CI Bootstrap for vllm-omni"
echo "Branch: ${BUILDKITE_BRANCH}"
echo "Commit: ${BUILDKITE_COMMIT}"
echo "AMD Hardware: ${AMD_MIRROR_HW}"
echo "CI Infra Branch: ${VLLM_CI_BRANCH}"

# Check that we're in the right directory
if [ ! -d ".buildkite" ]; then
    echo "Error: .buildkite directory not found. Please run this script from the repo root."
    exit 1
fi

# Install minijinja for template processing
echo "--- Installing minijinja"
if ! command -v minijinja-cli &> /dev/null; then
    curl -sSfL https://github.com/mitsuhiko/minijinja/releases/download/2.3.1/minijinja-cli-installer.sh | sh
    # Try both possible cargo env locations
    if [ -f "/var/lib/buildkite-agent/.cargo/env" ]; then
        source /var/lib/buildkite-agent/.cargo/env
    elif [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
else
    echo "minijinja-cli already installed"
fi

# Download AMD template from ci-infra
echo "--- Downloading AMD pipeline template"
curl -o .buildkite/test-template-amd.j2 \
    "https://raw.githubusercontent.com/vllm-project/ci-infra/${VLLM_CI_BRANCH}/buildkite/test-template-amd.j2?$(date +%s)"

# Check that test-amd.yaml exists
if [ ! -f ".buildkite/test-amd.yaml" ]; then
    echo "Error: .buildkite/test-amd.yaml not found!"
    exit 1
fi

# Get file diff for source_file_dependencies
get_diff() {
    git add . 2>/dev/null || true
    if git rev-parse origin/main >/dev/null 2>&1; then
        git diff --name-only --diff-filter=ACMDR $(git merge-base origin/main HEAD) 2>/dev/null || echo ""
    else
        # Fallback if origin/main doesn't exist
        git diff --name-only --diff-filter=ACMDR HEAD~1 2>/dev/null || echo ""
    fi
}

file_diff=$(get_diff)
if [[ $BUILDKITE_BRANCH == "main" ]]; then
    file_diff=$(git diff --name-only --diff-filter=ACMDR HEAD~1 2>/dev/null || echo "")
fi

# Convert file diff to pipe-separated list for jinja template
LIST_FILE_DIFF=$(echo "$file_diff" | tr '\n' '|' | tr ' ' '|' | sed 's/|$//')

echo "--- File Changes Detected"
if [ -n "$LIST_FILE_DIFF" ]; then
    echo "$file_diff"
else
    echo "No file changes detected (or unable to determine)"
fi

echo "--- Generating pipeline from test-amd.yaml"
echo "Run all tests: $RUN_ALL"

# Get merge base commit for reference
MERGE_BASE_COMMIT=$(git merge-base origin/main HEAD 2>/dev/null || git rev-parse HEAD~1 2>/dev/null || git rev-parse HEAD)

cd .buildkite

# Generate pipeline using minijinja
minijinja-cli test-template-amd.j2 test-amd.yaml \
    -D branch="$BUILDKITE_BRANCH" \
    -D list_file_diff="$LIST_FILE_DIFF" \
    -D run_all="$RUN_ALL" \
    -D nightly="$NIGHTLY" \
    -D mirror_hw="$AMD_MIRROR_HW" \
    -D fail_fast="false" \
    -D vllm_use_precompiled="0" \
    -D vllm_merge_base_commit="$MERGE_BASE_COMMIT" \
    -D cov_enabled="$COV_ENABLED" \
    -D vllm_ci_branch="$VLLM_CI_BRANCH" \
    | sed '/^[[:space:]]*$/d' \
    | sed '/queue:/a\    cluster: "CI"' \
    > pipeline.yaml

echo "--- Generated Pipeline Preview:"
cat pipeline.yaml

echo "--- Uploading pipeline to Buildkite"
buildkite-agent artifact upload pipeline.yaml
buildkite-agent pipeline upload pipeline.yaml

echo "--- Bootstrap complete!"
