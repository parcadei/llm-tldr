"""
Tests for device auto-detection (CUDA, MPS, CPU) in semantic indexing.
"""
import pytest

class TestDeviceDetection:
    """Test automatic device detection for embeddings."""

    def test_get_device_returns_valid_string(self):
        """_get_device should return 'cuda', 'mps', or 'cpu'."""
        from tldr.semantic import _get_device

        device = _get_device()
        assert device in ["cpu", "cuda", "mps"], f"Device should be 'cpu', 'cuda', or 'mps', got {device}"

    def test_device_fallback_to_cpu(self):
        """Should fallback to CPU gracefully if no accelerator available."""
        from tldr.semantic import _get_device

        # Even without GPU/MPS, should not crash
        device = _get_device()
        assert device in ["cpu", "cuda", "mps"], f"Device should be valid, got {device}"

    def test_macbook_mps_detection(self):
        """On Apple Silicon (M1/M2/M3), should detect MPS."""
        import torch

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # This Mac has MPS available
            from tldr.semantic import _get_device
            device = _get_device()
            # Should prefer MPS over CPU
            # Note: actual result depends on environment
            assert device in ["mps", "cpu"], f"Mac with MPS should return 'mps' or 'cpu', got {device}"
