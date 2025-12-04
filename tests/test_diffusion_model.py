"""
Test Z-Image-Turbo diffusion model on AMD GPU.

This test validates the Z-Image diffusion model can:
1. Load successfully on AMD GPU
2. Generate images with correct dimensions
3. Produce multiple outputs from batch generation

Model: Tongyi-MAI/Z-Image-Turbo
- Low-res (256x256) to avoid OOM on single GPU
- 2 inference steps for fast CI testing
"""

import pytest
import torch

from vllm_omni import Omni

models = ["Tongyi-MAI/Z-Image-Turbo"]


@pytest.mark.parametrize("model_name", models)
def test_diffusion_model(model_name: str):
    """Test Z-Image-Turbo diffusion model basic generation."""
    # Skip if no GPU available (shouldn't happen in CI, but good for local testing)
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU not available")

    # Initialize model
    m = Omni(model=model_name)

    # Low resolution to avoid OOM on single GPU
    height = 256
    width = 256

    images = m.generate(
        "a photo of a cat sitting on a laptop keyboard",
        height=height,
        width=width,
        num_inference_steps=2,  # Fast for CI
        guidance_scale=0.0,  # No CFG for speed
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=2,
    )

    # Validate outputs
    assert len(images) == 2, f"Expected 2 images, got {len(images)}"
    assert images[0].width == width, f"Image width is {images[0].width}, expected {width}"
    assert images[0].height == height, f"Image height is {images[0].height}, expected {height}"

    # Save sample output for verification
    images[0].save("z_image_output.png")
    print(f"Successfully generated {len(images)} images with Z-Image-Turbo")
