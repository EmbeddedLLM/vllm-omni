#!/bin/bash
# Explicit AMD CI bootstrap for vllm-omni
# Self-contained script that generates Buildkite pipeline for AMD GPU testing
#
# This script reads test-amd.yaml and generates a Buildkite pipeline
# with proper cluster configuration for the CI cluster.

set -euo pipefail

echo "=== vLLM-Omni AMD CI Bootstrap ==="
echo "Branch: ${BUILDKITE_BRANCH}"
echo "Commit: ${BUILDKITE_COMMIT}"
echo ""

# Check that we're in the right directory
if [ ! -d ".buildkite" ]; then
    echo "Error: .buildkite directory not found"
    echo "Please run this script from the repository root"
    exit 1
fi

if [ ! -f ".buildkite/test-amd.yaml" ]; then
    echo "Error: .buildkite/test-amd.yaml not found"
    exit 1
fi

echo "--- Generating AMD pipeline"

# Generate pipeline YAML explicitly
cat > .buildkite/pipeline.yaml << 'EOF'
steps:
  # AMD Build Step (placeholder - will build Docker image)
  - label: ":docker: Build AMD Docker Image"
    key: amd-build
    agents:
      queue: amd_mi325_1
      cluster: "CI"
    commands:
      - echo "AMD build step - image will be built by run-amd-test.sh"
      - echo "Image: rocm/vllm-omni-ci:${BUILDKITE_COMMIT}"
    timeout_in_minutes: 5

  # Z-Image Diffusion Model Test
  - label: ":rocm: Z-Image Diffusion Model Test"
    depends_on: amd-build
    agents:
      queue: amd_mi325_1
      cluster: "CI"
    command: bash .buildkite/scripts/hardware_ci/run-amd-test.sh "(command rocm-smi || true) && export VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1 && cd /vllm-omni-workspace/tests && pytest -v -s test_diffusion_model.py"
    env:
      HF_HOME: "/root/.cache/huggingface"
    timeout_in_minutes: 20
    retry:
      automatic:
        - exit_status: "*"
          limit: 1
EOF

echo "--- Generated Pipeline:"
cat .buildkite/pipeline.yaml

echo ""
echo "--- Uploading pipeline to Buildkite"
buildkite-agent pipeline upload .buildkite/pipeline.yaml

echo ""
echo "=== Bootstrap Complete ==="
