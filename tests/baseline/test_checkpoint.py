"""
Checkpoint 详细测试脚本
测试内容：
1. 基础保存/加载往返测试
2. DataParallel 兼容性测试
3. Optimizer 和 Scheduler 状态恢复
4. best.pth 自动保存
5. 目录自动创建
6. checkpoint 信息读取
"""

import os
import sys
import shutil
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

sys.path.insert(0, '/home/comp/25481568/code/HyperBody')

from models import UNet3D
from utils.checkpoint import save_checkpoint, load_checkpoint, get_checkpoint_info


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


class SimpleModel(nn.Module):
    """Simple model for quick testing"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=(2, 3, 4))
        return self.fc(x)


def test_basic_save_load():
    """Test basic save/load round-trip"""
    print_section("Test 1: Basic Save/Load Round-trip")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create model and optimizer
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Get original state
        original_params = {k: v.clone() for k, v in model.state_dict().items()}
        original_lr = optimizer.param_groups[0]['lr']

        print(f"  Original model conv weight sum: {original_params['conv.weight'].sum().item():.6f}")
        print(f"  Original optimizer LR: {original_lr}")

        # Modify model (simulate training)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Step optimizer and scheduler
        optimizer.step()
        scheduler.step()

        modified_params = {k: v.clone() for k, v in model.state_dict().items()}
        modified_lr = scheduler.get_last_lr()[0]

        print(f"  Modified model conv weight sum: {modified_params['conv.weight'].sum().item():.6f}")
        print(f"  Modified scheduler LR: {modified_lr}")

        # Save checkpoint
        state = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': 0.75,
        }
        save_path = save_checkpoint(state, tmpdir, 'test.pth')
        print(f"\n  Saved checkpoint to: {save_path}")
        print(f"  File exists: {os.path.exists(save_path)}")
        print(f"  File size: {os.path.getsize(save_path) / 1024:.2f} KB")

        # Create new model and load
        model2 = SimpleModel()
        optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
        scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=10)

        start_epoch, best_dice = load_checkpoint(
            save_path, model2, optimizer2, scheduler2
        )

        loaded_params = {k: v.clone() for k, v in model2.state_dict().items()}
        loaded_lr = scheduler2.get_last_lr()[0]

        print(f"\n  Loaded model conv weight sum: {loaded_params['conv.weight'].sum().item():.6f}")
        print(f"  Loaded scheduler LR: {loaded_lr}")
        print(f"  start_epoch: {start_epoch} (expected 6)")
        print(f"  best_dice: {best_dice} (expected 0.75)")

        # Verify
        for key in modified_params:
            diff = (modified_params[key] - loaded_params[key]).abs().max().item()
            assert diff < 1e-6, f"Parameter {key} mismatch: diff={diff}"

        assert start_epoch == 6
        assert best_dice == 0.75
        assert abs(loaded_lr - modified_lr) < 1e-8

        print("\n  ✓ Basic save/load test passed")


def test_dataparallel_compatibility():
    """Test DataParallel compatibility"""
    print_section("Test 2: DataParallel Compatibility")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case 2a: Save regular, load to DataParallel
        print("\n  Case 2a: Regular -> DataParallel")
        model = SimpleModel()
        with torch.no_grad():
            model.conv.weight.fill_(1.0)

        state = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'best_dice': 0.8,
        }
        save_checkpoint(state, tmpdir, 'regular.pth')

        # Load to DataParallel model (simulated, no actual GPU needed)
        model_dp = SimpleModel()
        model_dp = nn.DataParallel(model_dp)

        # Manually handle the loading (simulating GPU environment)
        checkpoint = torch.load(os.path.join(tmpdir, 'regular.pth'))
        model_state = checkpoint['model_state_dict']

        # Add 'module.' prefix
        model_state_dp = {f"module.{k}": v for k, v in model_state.items()}
        model_dp.load_state_dict(model_state_dp)

        assert model_dp.module.conv.weight.sum().item() == model.conv.weight.sum().item()
        print("    ✓ Regular -> DataParallel passed")

        # Case 2b: Save DataParallel, load to regular
        print("\n  Case 2b: DataParallel -> Regular")
        state_dp = {
            'epoch': 10,
            'model_state_dict': model_dp.state_dict(),  # Has 'module.' prefix
            'best_dice': 0.8,
        }
        save_checkpoint(state_dp, tmpdir, 'parallel.pth')

        model2 = SimpleModel()
        start_epoch, best_dice = load_checkpoint(
            os.path.join(tmpdir, 'parallel.pth'), model2
        )

        assert model2.conv.weight.sum().item() == model.conv.weight.sum().item()
        print("    ✓ DataParallel -> Regular passed")

        print("\n  ✓ DataParallel compatibility test passed")


def test_best_checkpoint():
    """Test best checkpoint saving"""
    print_section("Test 3: Best Checkpoint Auto-save")

    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()

        # Save with is_best=False
        state1 = {'epoch': 1, 'model_state_dict': model.state_dict(), 'best_dice': 0.5}
        save_checkpoint(state1, tmpdir, 'epoch_1.pth', is_best=False)

        print(f"  After epoch 1 (is_best=False):")
        print(f"    epoch_1.pth exists: {os.path.exists(os.path.join(tmpdir, 'epoch_1.pth'))}")
        print(f"    best.pth exists: {os.path.exists(os.path.join(tmpdir, 'best.pth'))}")

        # Save with is_best=True
        state2 = {'epoch': 2, 'model_state_dict': model.state_dict(), 'best_dice': 0.7}
        save_checkpoint(state2, tmpdir, 'epoch_2.pth', is_best=True)

        print(f"\n  After epoch 2 (is_best=True):")
        print(f"    epoch_2.pth exists: {os.path.exists(os.path.join(tmpdir, 'epoch_2.pth'))}")
        print(f"    best.pth exists: {os.path.exists(os.path.join(tmpdir, 'best.pth'))}")

        # Verify best.pth has correct content
        info = get_checkpoint_info(os.path.join(tmpdir, 'best.pth'))
        print(f"\n  best.pth content:")
        print(f"    epoch: {info['epoch']}")
        print(f"    best_dice: {info['best_dice']}")

        assert info['epoch'] == 2
        assert info['best_dice'] == 0.7

        print("\n  ✓ Best checkpoint test passed")


def test_directory_creation():
    """Test automatic directory creation"""
    print_section("Test 4: Directory Auto-creation")

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = os.path.join(tmpdir, 'level1', 'level2', 'checkpoints')

        print(f"  Target directory: {nested_dir}")
        print(f"  Exists before save: {os.path.exists(nested_dir)}")

        model = SimpleModel()
        state = {'epoch': 1, 'model_state_dict': model.state_dict(), 'best_dice': 0.5}
        save_path = save_checkpoint(state, nested_dir, 'test.pth')

        print(f"  Exists after save: {os.path.exists(nested_dir)}")
        print(f"  Checkpoint saved: {os.path.exists(save_path)}")

        assert os.path.exists(nested_dir)
        assert os.path.exists(save_path)

        print("\n  ✓ Directory creation test passed")


def test_checkpoint_info():
    """Test checkpoint info extraction"""
    print_section("Test 5: Checkpoint Info Extraction")

    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()

        state = {
            'epoch': 42,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {'test': 'data'},
            'scheduler_state_dict': {'test': 'data'},
            'best_dice': 0.8765,
            'config': {'lr': 0.001, 'batch_size': 4},
        }
        save_path = save_checkpoint(state, tmpdir, 'full.pth')

        info = get_checkpoint_info(save_path)

        print(f"  Checkpoint info:")
        print(f"    epoch: {info['epoch']}")
        print(f"    best_dice: {info['best_dice']}")
        print(f"    config: {info.get('config', 'N/A')}")

        assert info['epoch'] == 42
        assert info['best_dice'] == 0.8765
        assert info['config']['lr'] == 0.001

        print("\n  ✓ Checkpoint info test passed")


def test_with_real_model():
    """Test with actual UNet3D model"""
    print_section("Test 6: Real UNet3D Model")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Creating UNet3D model...")
        model = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters: {num_params:,}")

        # Simulate training modifications
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        # Get checksum of parameters
        original_checksum = sum(p.sum().item() for p in model.parameters())
        print(f"  Original param checksum: {original_checksum:.4f}")

        # Save checkpoint
        state = {
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': 0.65,
        }
        save_path = save_checkpoint(state, tmpdir, 'unet3d.pth')

        file_size = os.path.getsize(save_path) / (1024 * 1024)
        print(f"  Checkpoint size: {file_size:.2f} MB")

        # Load into new model
        model2 = UNet3D(in_channels=1, num_classes=70, base_channels=32)
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
        scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, factor=0.5, patience=10)

        start_epoch, best_dice = load_checkpoint(
            save_path, model2, optimizer2, scheduler2
        )

        loaded_checksum = sum(p.sum().item() for p in model2.parameters())
        print(f"  Loaded param checksum:  {loaded_checksum:.4f}")
        print(f"  start_epoch: {start_epoch}")
        print(f"  best_dice: {best_dice}")

        # Verify
        assert abs(original_checksum - loaded_checksum) < 1e-4
        assert start_epoch == 51
        assert best_dice == 0.65

        print("\n  ✓ Real UNet3D model test passed")


def test_error_handling():
    """Test error handling"""
    print_section("Test 7: Error Handling")

    model = SimpleModel()

    # Test loading non-existent file
    print("  Testing FileNotFoundError...")
    try:
        load_checkpoint('/nonexistent/path/checkpoint.pth', model)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"    Caught expected error: {type(e).__name__}")
        print("    ✓ FileNotFoundError handled correctly")

    # Test get_checkpoint_info with non-existent file
    print("\n  Testing get_checkpoint_info error...")
    try:
        get_checkpoint_info('/nonexistent/path/checkpoint.pth')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"    Caught expected error: {type(e).__name__}")
        print("    ✓ Error handled correctly")

    print("\n  ✓ Error handling test passed")


def test_cosine_annealing_checkpoint():
    """Test CosineAnnealingLR state_dict save/load round-trip"""
    print_section("Test 8: CosineAnnealingLR Checkpoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

        # Simulate 15 epochs of training
        for _ in range(15):
            optimizer.step()
            scheduler.step()

        lr_before = optimizer.param_groups[0]['lr']
        print(f"  LR after 15 steps: {lr_before:.8f}")

        # Save checkpoint
        state = {
            'epoch': 14,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': 0.6,
        }
        save_path = save_checkpoint(state, tmpdir, 'cosine.pth')

        # Load into fresh model
        model2 = SimpleModel()
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=50, eta_min=1e-6)

        start_epoch, best_dice = load_checkpoint(
            save_path, model2, optimizer2, scheduler2
        )

        lr_after = optimizer2.param_groups[0]['lr']
        print(f"  LR after load:     {lr_after:.8f}")
        print(f"  start_epoch: {start_epoch} (expected 15)")

        assert start_epoch == 15
        assert abs(lr_before - lr_after) < 1e-10, f"LR mismatch: {lr_before} vs {lr_after}"

        # Step both schedulers one more time and verify they stay in sync
        scheduler.step()
        scheduler2.step()
        lr_orig_next = optimizer.param_groups[0]['lr']
        lr_loaded_next = optimizer2.param_groups[0]['lr']
        print(f"  LR after one more step (original):  {lr_orig_next:.8f}")
        print(f"  LR after one more step (loaded):    {lr_loaded_next:.8f}")
        assert abs(lr_orig_next - lr_loaded_next) < 1e-10

        print("\n  ✓ CosineAnnealingLR checkpoint test passed")


def test_sequential_lr_checkpoint():
    """Test SequentialLR(LinearLR + CosineAnnealingLR) save/load round-trip"""
    print_section("Test 9: SequentialLR Checkpoint (warmup + cosine)")

    with tempfile.TemporaryDirectory() as tmpdir:
        total_epochs = 50
        warmup_epochs = 5

        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )

        # Step through warmup phase and into cosine phase (8 epochs)
        lrs = []
        for i in range(8):
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)
            optimizer.step()
            scheduler.step()

        lr_before = optimizer.param_groups[0]['lr']
        print(f"  LR trajectory (8 steps): {[f'{x:.6f}' for x in lrs]}")
        print(f"  LR at save point: {lr_before:.8f}")

        # Verify warmup happened (LR should increase during first 5 epochs)
        assert lrs[0] < lrs[3], f"Warmup not working: lr[0]={lrs[0]} >= lr[3]={lrs[3]}"
        # Verify cosine started (LR should decrease after warmup)
        assert lrs[6] < lrs[5] or lrs[7] < lrs[5], "Cosine decay not started after warmup"

        # Save checkpoint
        state = {
            'epoch': 7,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': 0.55,
        }
        save_path = save_checkpoint(state, tmpdir, 'sequential.pth')

        # Load into fresh model with identical scheduler setup
        model2 = SimpleModel()
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)

        cosine_sched2 = CosineAnnealingLR(
            optimizer2, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        warmup_sched2 = LinearLR(
            optimizer2, start_factor=1e-3, total_iters=warmup_epochs
        )
        scheduler2 = SequentialLR(
            optimizer2, [warmup_sched2, cosine_sched2], milestones=[warmup_epochs]
        )

        start_epoch, best_dice = load_checkpoint(
            save_path, model2, optimizer2, scheduler2
        )

        lr_after = optimizer2.param_groups[0]['lr']
        print(f"  LR after load:     {lr_after:.8f}")

        assert start_epoch == 8
        assert abs(lr_before - lr_after) < 1e-10, f"LR mismatch: {lr_before} vs {lr_after}"

        # Step both and verify sync
        scheduler.step()
        scheduler2.step()
        lr_orig_next = optimizer.param_groups[0]['lr']
        lr_loaded_next = optimizer2.param_groups[0]['lr']
        print(f"  LR next step (original): {lr_orig_next:.8f}")
        print(f"  LR next step (loaded):   {lr_loaded_next:.8f}")
        assert abs(lr_orig_next - lr_loaded_next) < 1e-10

        print("\n  ✓ SequentialLR checkpoint test passed")


def test_cosine_multi_param_group():
    """Test CosineAnnealingLR with 2 param groups (visual + text, matching train.py)"""
    print_section("Test 10: Cosine Multi Param Group Checkpoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        total_epochs = 50
        warmup_epochs = 5

        model = SimpleModel()
        # 2 param groups: visual (lr=1e-3) and text (lr=1e-5)
        optimizer = optim.Adam([
            {'params': [model.conv.weight, model.conv.bias], 'lr': 1e-3},
            {'params': [model.fc.weight, model.fc.bias], 'lr': 1e-5},
        ], weight_decay=1e-5)

        cosine_sched = CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-3, total_iters=warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )

        # Step through 12 epochs
        for _ in range(12):
            optimizer.step()
            scheduler.step()

        lr_visual_before = optimizer.param_groups[0]['lr']
        lr_text_before = optimizer.param_groups[1]['lr']
        print(f"  Visual LR at save: {lr_visual_before:.8f}")
        print(f"  Text LR at save:   {lr_text_before:.8f}")

        # Save
        state = {
            'epoch': 11,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': 0.5,
        }
        save_path = save_checkpoint(state, tmpdir, 'multi_group.pth')

        # Load into fresh setup
        model2 = SimpleModel()
        optimizer2 = optim.Adam([
            {'params': [model2.conv.weight, model2.conv.bias], 'lr': 1e-3},
            {'params': [model2.fc.weight, model2.fc.bias], 'lr': 1e-5},
        ], weight_decay=1e-5)

        cosine_sched2 = CosineAnnealingLR(
            optimizer2, T_max=total_epochs - warmup_epochs, eta_min=1e-6
        )
        warmup_sched2 = LinearLR(
            optimizer2, start_factor=1e-3, total_iters=warmup_epochs
        )
        scheduler2 = SequentialLR(
            optimizer2, [warmup_sched2, cosine_sched2], milestones=[warmup_epochs]
        )

        start_epoch, best_dice = load_checkpoint(
            save_path, model2, optimizer2, scheduler2
        )

        lr_visual_after = optimizer2.param_groups[0]['lr']
        lr_text_after = optimizer2.param_groups[1]['lr']
        print(f"  Visual LR after load: {lr_visual_after:.8f}")
        print(f"  Text LR after load:   {lr_text_after:.8f}")

        assert start_epoch == 12
        assert abs(lr_visual_before - lr_visual_after) < 1e-10, \
            f"Visual LR mismatch: {lr_visual_before} vs {lr_visual_after}"
        assert abs(lr_text_before - lr_text_after) < 1e-10, \
            f"Text LR mismatch: {lr_text_before} vs {lr_text_after}"

        # Verify both groups stay in sync after another step
        scheduler.step()
        scheduler2.step()
        assert abs(optimizer.param_groups[0]['lr'] - optimizer2.param_groups[0]['lr']) < 1e-10
        assert abs(optimizer.param_groups[1]['lr'] - optimizer2.param_groups[1]['lr']) < 1e-10

        print("\n  ✓ Cosine multi param group checkpoint test passed")


def main():
    print("\n" + "="*60)
    print("  Checkpoint 详细测试")
    print("="*60)

    test_basic_save_load()
    test_dataparallel_compatibility()
    test_best_checkpoint()
    test_directory_creation()
    test_checkpoint_info()
    test_with_real_model()
    test_error_handling()
    test_cosine_annealing_checkpoint()
    test_sequential_lr_checkpoint()
    test_cosine_multi_param_group()

    print("\n" + "="*60)
    print("  All 10 tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
