"""
Setup script for vLLM-Omni with hardware-dependent installation.

This setup.py implements platform-aware dependency routing so users can run
`pip install vllm-omni` and automatically receive the correct platform-specific
dependencies (CUDA/ROCm/CPU/XPU/NPU) without requiring extras like `[cuda]`.
"""

import os
import sys
from pathlib import Path
from typing import List

from setuptools import setup


def detect_target_device() -> str:
    """
    Detect the target device for installation following RFC priority rules.

    Priority order:
    1. VLLM_OMNI_TARGET_DEVICE environment variable (highest priority)
    2. Torch backend detection (cuda, rocm, npu, xpu)
    3. CPU fallback (default)

    Returns:
        str: Device name ('cuda', 'rocm', 'npu', 'xpu', or 'cpu')
    """
    # Priority 1: Explicit override via environment variable
    target_device = os.environ.get("VLLM_OMNI_TARGET_DEVICE")
    if target_device:
        valid_devices = ["cuda", "rocm", "npu", "xpu", "cpu"]
        if target_device.lower() in valid_devices:
            print(f"Using target device from VLLM_OMNI_TARGET_DEVICE: {target_device.lower()}")
            return target_device.lower()
        else:
            print(f"Warning: Invalid VLLM_OMNI_TARGET_DEVICE '{target_device}', falling back to auto-detection")

    # Priority 2: Torch backend detection
    try:
        import torch

        # Check for CUDA
        if torch.version.cuda is not None:
            print("Detected CUDA backend from torch")
            return "cuda"

        # Check for ROCm (AMD)
        if torch.version.hip is not None:
            print("Detected ROCm backend from torch")
            return "rocm"

        # Check for NPU (Ascend)
        if hasattr(torch, 'npu'):
            try:
                if torch.npu.is_available():
                    print("Detected NPU backend from torch")
                    return "npu"
            except Exception:
                pass

        # Check for XPU (Intel)
        if hasattr(torch, 'xpu'):
            try:
                if torch.xpu.is_available():
                    print("Detected XPU backend from torch")
                    return "xpu"
            except Exception:
                pass

        print("No GPU backend detected in torch, defaulting to CPU")
        return "cpu"

    except ImportError:
        print("PyTorch not found, defaulting to CPU installation")
        return "cpu"


def load_requirements(file_path: Path) -> List[str]:
    """
    Load requirements from a file, supporting -r directive for recursive loading.

    Args:
        file_path: Path to the requirements file

    Returns:
        List of requirement strings
    """
    requirements = []

    if not file_path.exists():
        print(f"Warning: Requirements file not found: {file_path}")
        return requirements

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Handle -r directive for recursive loading
            if line.startswith('-r '):
                nested_file = line[3:].strip()
                nested_path = file_path.parent / nested_file
                requirements.extend(load_requirements(nested_path))
            else:
                requirements.append(line)

    return requirements


def get_install_requires() -> List[str]:
    """
    Get the list of dependencies based on detected platform.

    Returns:
        List of requirement strings for the detected platform
    """
    device = detect_target_device()
    requirements_dir = Path(__file__).parent / "requirements"
    requirements_file = requirements_dir / f"{device}.txt"

    print(f"Loading requirements from: {requirements_file}")
    requirements = load_requirements(requirements_file)

    if not requirements:
        print(f"Warning: No requirements loaded for device '{device}'")
    else:
        print(f"Loaded {len(requirements)} requirements for {device}")

    return requirements


if __name__ == "__main__":
    # Get platform-specific dependencies
    install_requires = get_install_requires()

    # Setup configuration
    setup(
        install_requires=install_requires,
    )
