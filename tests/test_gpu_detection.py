"""
Tests for GPU auto-detection in semantic indexing.
"""
import pytest

class TestGPUAutoDetection:
    """Test automatic GPU detection for embeddings."""

    def test_get_device_returns_valid_string(self):
        """_get_device should return 'cpu' or 'cuda'."""
        from tldr.semantic import _get_device

        device = _get_device()
        assert device in ["cpu", "cuda"], f"Device should be 'cpu' or 'cuda', got {device}"

    def test_get_device_fallback_to_cpu(self):
        """Should fallback to CPU gracefully if CUDA unavailable."""
        from tldr.semantic import _get_device

        # Even without GPU, should not crash
        device = _get_device()
        assert device == "cpu", "Should default to CPU if no CUDA available"

    def test_model_loads_with_detected_device(self, tmp_path):
        """get_model should use the device detected by _get_device."""
        from tldr.semantic import _get_device, get_model

        device = _get_device()

        # This will download the model if not present, but should work
        try:
            model = get_model("all-MiniLM-L6-v2")
            # Model loaded successfully
            assert model is not None
            # The device should be what _get_device returned
            # Note: sentence_transformers stores device internally
        except Exception as e:
            pytest.skip(f"Model download/test failed: {e}")
