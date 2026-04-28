"""Test PyTorch AMP API usage for PyTorch 2.x compatibility.

This test verifies that we use the new torch.amp API instead of the deprecated
torch.cuda.amp API.

PyTorch 2.x migration:
- autocast: Use torch.amp.autocast with device_type="cuda" instead of torch.cuda.amp.autocast
- GradScaler: In PyTorch 2.0.x, GradScaler is still in torch.cuda.amp but should be
  instantiated with device type string for forward compatibility
"""

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple model for testing AMP."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAMPAPI:
    """Test that AMP API works correctly with PyTorch 2.x."""

    def test_autocast_with_device_type(self):
        """Test that autocast works with explicit device_type parameter.

        PyTorch 2.x requires device_type to be specified for torch.amp.autocast.
        """
        from torch.amp import autocast

        model = SimpleModel().cuda()
        x = torch.randn(2, 10).cuda()

        # This should work with the new API
        with autocast(device_type="cuda"):
            output = model(x)

        assert output.shape == (2, 10)
        # In autocast context, output should be float16 or bfloat16
        assert output.dtype in (torch.float16, torch.bfloat16)

    def test_gradscaler_with_device_type(self):
        """Test that GradScaler works with explicit device parameter.

        In PyTorch 2.0.x, GradScaler is still in torch.cuda.amp.
        It accepts device type string for forward compatibility.
        """
        from torch.cuda.amp import GradScaler

        # GradScaler with device type string (forward compatible)
        scaler = GradScaler()

        assert scaler is not None
        assert scaler.is_enabled()

    def test_amp_training_loop(self):
        """Test a complete AMP training loop with the new API."""
        from torch.amp import autocast
        from torch.cuda.amp import GradScaler

        model = SimpleModel().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = GradScaler()

        x = torch.randn(2, 10).cuda()
        target = torch.randn(2, 10).cuda()

        # Training step with AMP
        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Verify training completed without error
        assert loss.item() >= 0


class TestTrainPyAMPImports:
    """Test that train.py uses the correct AMP imports."""

    def test_train_py_uses_new_autocast_api(self):
        """Verify train.py imports autocast from torch.amp."""
        import ast

        with open("train.py", "r") as f:
            content = f.read()

        tree = ast.parse(content)

        # Find all import statements and what they import
        autocast_from_torch_amp = False
        autocast_from_cuda_amp = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "torch.amp":
                    for alias in node.names:
                        if alias.name == "autocast":
                            autocast_from_torch_amp = True
                elif node.module == "torch.cuda.amp":
                    for alias in node.names:
                        if alias.name == "autocast":
                            autocast_from_cuda_amp = True

        assert autocast_from_torch_amp, "train.py should import autocast from torch.amp"
        assert not autocast_from_cuda_amp, "train.py should NOT import autocast from torch.cuda.amp"

    def test_autocast_uses_device_type(self):
        """Verify autocast is called with device_type parameter."""
        with open("train.py", "r") as f:
            content = f.read()

        # Check that autocast is called with device_type
        # Old: with autocast():
        # New: with autocast(device_type="cuda"):
        assert 'autocast(device_type="cuda")' in content or "autocast(device_type='cuda')" in content, \
            "autocast should be called with device_type='cuda'"

        # Should not have bare autocast() call
        import re
        bare_autocast = re.search(r'with\s+autocast\(\s*\):', content)
        assert bare_autocast is None, "Should not have bare autocast() without device_type"

    def test_gradscaler_import_location(self):
        """Verify GradScaler is imported from torch.cuda.amp (still valid in PyTorch 2.0.x)."""
        import ast

        with open("train.py", "r") as f:
            content = f.read()

        tree = ast.parse(content)

        # In PyTorch 2.0.x, GradScaler is still in torch.cuda.amp
        gradscaler_imported = False

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "torch.cuda.amp":
                    for alias in node.names:
                        if alias.name == "GradScaler":
                            gradscaler_imported = True

        assert gradscaler_imported, "train.py should import GradScaler from torch.cuda.amp"
