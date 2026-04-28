"""
Model unit tests for DenseBlock, UNet3D, and Losses
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/home/comp/25481568/code/HyperBody')

from models.dense_block import DenseLayer, DenseBlock
from models.unet3d import UNet3D, ConvBlock, Encoder, Decoder
from models.losses import DiceLoss, CombinedLoss


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_dense_layer():
    """Test DenseLayer"""
    print_section("Test 1: DenseLayer")

    layer = DenseLayer(in_channels=64, growth_rate=32, bn_size=4)

    x = torch.randn(2, 64, 8, 8, 8)
    out = layer(x)

    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Expected: (2, 32, 8, 8, 8)")

    assert out.shape == (2, 32, 8, 8, 8)
    print("  OK")


def test_dense_block():
    """Test DenseBlock"""
    print_section("Test 2: DenseBlock")

    block = DenseBlock(in_channels=256, num_layers=4, growth_rate=32, bn_size=4)

    x = torch.randn(1, 256, 16, 12, 32)
    out = block(x)

    expected_channels = 256 + 4 * 32  # 384

    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Expected channels: {expected_channels}")
    print(f"  out_channels attr: {block.out_channels}")

    assert out.shape == (1, 384, 16, 12, 32)
    assert block.out_channels == 384
    print("  OK")


def test_conv_block():
    """Test ConvBlock"""
    print_section("Test 3: ConvBlock")

    block = ConvBlock(in_channels=32, out_channels=64)

    x = torch.randn(2, 32, 16, 16, 16)
    out = block(x)

    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")

    assert out.shape == (2, 64, 16, 16, 16)
    print("  OK")


def test_encoder():
    """Test Encoder"""
    print_section("Test 4: Encoder")

    encoder = Encoder(in_channels=32, out_channels=64)

    x = torch.randn(2, 32, 16, 16, 16)
    out = encoder(x)

    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Expected: (2, 64, 8, 8, 8) [spatial dims halved]")

    assert out.shape == (2, 64, 8, 8, 8)
    print("  OK")


def test_decoder():
    """Test Decoder"""
    print_section("Test 5: Decoder")

    decoder = Decoder(in_channels=128, out_channels=64)

    x = torch.randn(2, 64, 8, 8, 8)  # From previous level
    skip = torch.randn(2, 64, 16, 16, 16)  # Skip connection

    out = decoder(x, skip)

    print(f"  Input x:    {tuple(x.shape)}")
    print(f"  Input skip: {tuple(skip.shape)}")
    print(f"  Output:     {tuple(out.shape)}")
    print(f"  Expected: (2, 64, 16, 16, 16)")

    assert out.shape == (2, 64, 16, 16, 16)
    print("  OK")


def test_unet3d_forward():
    """Test UNet3D forward pass"""
    print_section("Test 6: UNet3D Forward")

    model = UNet3D(
        in_channels=1,
        num_classes=70,
        base_channels=32,
        growth_rate=32,
        dense_layers=4,
    )

    # Use smaller volume for faster test
    x = torch.randn(1, 1, 64, 48, 128)
    out = model(x)

    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Expected: (1, 70, 64, 48, 128)")

    assert out.shape == (1, 70, 64, 48, 128)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    print("  OK")


def test_unet3d_backward():
    """Test UNet3D backward pass"""
    print_section("Test 7: UNet3D Backward")

    model = UNet3D(in_channels=1, num_classes=10, base_channels=16)

    x = torch.randn(1, 1, 32, 32, 32)
    targets = torch.randint(0, 10, (1, 32, 32, 32))

    out = model(x)
    loss = nn.CrossEntropyLoss()(out, targets)

    print(f"  Loss: {loss.item():.4f}")

    loss.backward()

    # Check all parameters have gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())

    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")

    assert params_with_grad == total_params
    print("  OK")


def test_dice_loss():
    """Test DiceLoss"""
    print_section("Test 8: DiceLoss")

    loss_fn = DiceLoss(smooth=1.0)

    logits = torch.randn(2, 10, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 10, (2, 8, 8, 8))

    loss = loss_fn(logits, targets)

    print(f"  Logits:  {tuple(logits.shape)}")
    print(f"  Targets: {tuple(targets.shape)}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Loss in [0, 1]: {0 <= loss.item() <= 1}")

    assert 0 <= loss.item() <= 1
    assert loss.shape == ()

    loss.backward()
    assert logits.grad is not None
    print("  OK")


def test_dice_loss_perfect():
    """Test DiceLoss with perfect prediction"""
    print_section("Test 9: DiceLoss Perfect Prediction")

    loss_fn = DiceLoss(smooth=1e-5)

    # Create perfect prediction
    targets = torch.randint(0, 5, (2, 8, 8, 8))
    logits = torch.full((2, 5, 8, 8, 8), -10.0)

    for b in range(2):
        for d in range(8):
            for h in range(8):
                for w in range(8):
                    c = targets[b, d, h, w].item()
                    logits[b, c, d, h, w] = 10.0

    loss = loss_fn(logits, targets)

    print(f"  Loss: {loss.item():.6f}")
    print(f"  Expected: ~0 (perfect prediction)")

    assert loss.item() < 0.01
    print("  OK")


def test_combined_loss():
    """Test CombinedLoss"""
    print_section("Test 10: CombinedLoss")

    loss_fn = CombinedLoss(num_classes=10, ce_weight=0.5, dice_weight=0.5)

    logits = torch.randn(2, 10, 8, 8, 8, requires_grad=True)
    targets = torch.randint(0, 10, (2, 8, 8, 8))

    loss = loss_fn(logits, targets)

    print(f"  Loss: {loss.item():.4f}")

    loss.backward()
    assert logits.grad is not None
    print("  OK")


def test_combined_loss_with_weights():
    """Test CombinedLoss with class weights"""
    print_section("Test 11: CombinedLoss with Class Weights")

    class_weights = torch.ones(10)
    class_weights[0] = 0.1  # Lower weight for background

    loss_fn = CombinedLoss(
        num_classes=10,
        ce_weight=0.5,
        dice_weight=0.5,
        class_weights=class_weights
    )

    logits = torch.randn(2, 10, 8, 8, 8)
    targets = torch.randint(0, 10, (2, 8, 8, 8))

    loss = loss_fn(logits, targets)

    print(f"  Loss: {loss.item():.4f}")
    print("  OK")


def main():
    print("\n" + "="*60)
    print("  Model Unit Tests")
    print("="*60)

    test_dense_layer()
    test_dense_block()
    test_conv_block()
    test_encoder()
    test_decoder()
    test_unet3d_forward()
    test_unet3d_backward()
    test_dice_loss()
    test_dice_loss_perfect()
    test_combined_loss()
    test_combined_loss_with_weights()

    print("\n" + "="*60)
    print("  All model tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
