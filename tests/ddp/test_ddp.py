"""
Tests for DistributedDataParallel (DDP) helper functions.

Tests cover:
1. is_distributed() - detects distributed mode
2. get_rank() - returns process rank
3. get_world_size() - returns total processes
4. is_main_process() - checks if rank 0
5. setup_distributed() - initializes DDP
6. cleanup_distributed() - cleanup DDP
7. DiceMetric.sync_across_processes() - sync metrics across ranks
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, '/home/comp/25481568/code/HyperBody')


class TestDDPHelperFunctions(unittest.TestCase):
    """Test DDP helper functions in train.py"""

    def test_is_distributed_returns_false_when_not_initialized(self):
        """is_distributed() should return False when dist is not initialized"""
        from train import is_distributed

        # When not in distributed mode, should return False
        result = is_distributed()
        self.assertFalse(result)

    def test_get_rank_returns_zero_when_not_distributed(self):
        """get_rank() should return 0 when not in distributed mode"""
        from train import get_rank

        result = get_rank()
        self.assertEqual(result, 0)

    def test_get_world_size_returns_one_when_not_distributed(self):
        """get_world_size() should return 1 when not in distributed mode"""
        from train import get_world_size

        result = get_world_size()
        self.assertEqual(result, 1)

    def test_is_main_process_returns_true_when_not_distributed(self):
        """is_main_process() should return True when not in distributed mode"""
        from train import is_main_process

        result = is_main_process()
        self.assertTrue(result)

    def test_setup_distributed_sets_cuda_visible_devices(self):
        """setup_distributed() should set CUDA_VISIBLE_DEVICES if gpu_ids provided"""
        from train import setup_distributed

        # Clear env var if exists
        old_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

        # Also ensure WORLD_SIZE is not set (non-distributed)
        old_world_size = os.environ.get("WORLD_SIZE")
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]

        try:
            local_rank = setup_distributed(gpu_ids=[2, 3])

            # Should set CUDA_VISIBLE_DEVICES
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "2,3")
            # Should return 0 when not in torchrun
            self.assertEqual(local_rank, 0)
        finally:
            # Restore env
            if old_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_env
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            if old_world_size is not None:
                os.environ["WORLD_SIZE"] = old_world_size

    def test_setup_distributed_does_not_override_existing_cuda_visible_devices(self):
        """setup_distributed() should NOT override existing CUDA_VISIBLE_DEVICES"""
        from train import setup_distributed

        old_env = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        old_world_size = os.environ.get("WORLD_SIZE")
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]

        try:
            setup_distributed(gpu_ids=[5, 6])

            # Should NOT change because CUDA_VISIBLE_DEVICES was already set
            self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "0,1")
        finally:
            if old_env is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_env
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            if old_world_size is not None:
                os.environ["WORLD_SIZE"] = old_world_size

    def test_cleanup_distributed_runs_without_error_when_not_distributed(self):
        """cleanup_distributed() should run without error when not in distributed mode"""
        from train import cleanup_distributed

        # Should not raise any error
        cleanup_distributed()


class TestSetupLoggingWithMainProcess(unittest.TestCase):
    """Test setup_logging with is_main parameter"""

    def test_setup_logging_with_is_main_true_creates_file_handler(self):
        """setup_logging(is_main=True) should create file and stream handlers"""
        from train import setup_logging
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Reset logging config
            logging.root.handlers = []

            logger = setup_logging(tmpdir, is_main=True)

            # Check log file was created
            log_file = os.path.join(tmpdir, "training.log")
            self.assertTrue(os.path.exists(log_file))

    def test_setup_logging_with_is_main_false_uses_null_handler(self):
        """setup_logging(is_main=False) should use NullHandler only"""
        from train import setup_logging
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            # Reset logging config
            logging.root.handlers = []

            logger = setup_logging(tmpdir, is_main=False)

            # Should have only NullHandler
            has_null_handler = any(
                isinstance(h, logging.NullHandler)
                for h in logging.root.handlers
            )
            self.assertTrue(has_null_handler)

            # Should NOT have FileHandler or StreamHandler
            has_file_handler = any(
                isinstance(h, logging.FileHandler)
                for h in logging.root.handlers
            )
            has_stream_handler = any(
                isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                for h in logging.root.handlers
            )
            self.assertFalse(has_file_handler)
            self.assertFalse(has_stream_handler)


class TestTrainOneEpochWithDistributedSampler(unittest.TestCase):
    """Test train_one_epoch with DistributedSampler"""

    def test_train_one_epoch_calls_set_epoch_on_distributed_sampler(self):
        """train_one_epoch() should call sampler.set_epoch() if sampler has that method"""
        from train import train_one_epoch
        from torch.utils.data import DataLoader, TensorDataset
        from torch.utils.data.distributed import DistributedSampler

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 2, 1)

            def forward(self, x):
                return self.conv(x)

        model = SimpleModel()
        device = torch.device("cpu")

        # Create mock sampler with set_epoch
        mock_sampler = mock.MagicMock()
        mock_sampler.set_epoch = mock.MagicMock()

        # Create simple dataset and loader
        inputs = torch.randn(4, 1, 4, 4, 4)
        targets = torch.randint(0, 2, (4, 4, 4, 4))
        dataset = TensorDataset(inputs, targets)

        # Create loader with mock sampler
        loader = DataLoader(dataset, batch_size=2, sampler=mock_sampler)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Run one epoch
        train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0, epoch=5)

        # Check set_epoch was called with the correct epoch
        mock_sampler.set_epoch.assert_called_once_with(5)


class TestDiceMetricSyncAcrossProcesses(unittest.TestCase):
    """Test DiceMetric.sync_across_processes()"""

    def test_sync_across_processes_method_exists(self):
        """DiceMetric should have sync_across_processes method"""
        from utils.metrics import DiceMetric

        metric = DiceMetric(num_classes=10)
        self.assertTrue(hasattr(metric, 'sync_across_processes'))
        self.assertTrue(callable(getattr(metric, 'sync_across_processes')))

    def test_sync_across_processes_does_nothing_when_accumulators_none(self):
        """sync_across_processes() should return early if accumulators are None"""
        from utils.metrics import DiceMetric

        metric = DiceMetric(num_classes=10)
        # Before any update, _intersection is None
        self.assertIsNone(metric._intersection)

        # Should not raise any error
        metric.sync_across_processes()

    def test_sync_across_processes_calls_all_reduce_on_accumulators(self):
        """sync_across_processes() should call dist.all_reduce on accumulators"""
        from utils.metrics import DiceMetric

        metric = DiceMetric(num_classes=5)

        # Initialize accumulators by doing a fake update
        logits = torch.randn(1, 5, 4, 4, 4)
        targets = torch.randint(0, 5, (1, 4, 4, 4))
        metric.update(logits, targets)

        # Now mock dist.all_reduce
        with mock.patch('torch.distributed.all_reduce') as mock_all_reduce:
            # Need to also mock dist.is_available and dist.is_initialized
            with mock.patch('torch.distributed.is_available', return_value=True):
                with mock.patch('torch.distributed.is_initialized', return_value=True):
                    metric.sync_across_processes()

                    # all_reduce should be called 3 times: intersection, pred_sum, target_sum
                    self.assertEqual(mock_all_reduce.call_count, 3)


class TestCheckpointDDPHandling(unittest.TestCase):
    """Test checkpoint.py handling of DDP wrapped models"""

    def test_load_checkpoint_detects_wrapped_model_via_hasattr(self):
        """load_checkpoint() should detect wrapped models using hasattr(model, 'module')"""
        import tempfile
        from utils.checkpoint import save_checkpoint, load_checkpoint

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 2, 1)

            def forward(self, x):
                return self.conv(x)

        # Create a wrapper that has 'module' attribute (like DDP does)
        # but is NOT an instance of nn.DataParallel
        class NonDataParallelWrapper(nn.Module):
            """Wrapper that has 'module' but is not DataParallel"""
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a regular model checkpoint
            model = SimpleModel()
            with torch.no_grad():
                model.conv.weight.fill_(3.0)
                model.conv.bias.fill_(1.5)

            state = {
                'epoch': 20,
                'model_state_dict': model.state_dict(),  # No 'module.' prefix
                'best_dice': 0.9,
            }
            save_path = os.path.join(tmpdir, 'regular.pth')
            save_checkpoint(state, tmpdir, 'regular.pth')

            # Create wrapped model (not DataParallel, but has .module attribute)
            model2 = SimpleModel()
            wrapped_model = NonDataParallelWrapper(model2)

            # Verify it's not a DataParallel
            self.assertFalse(isinstance(wrapped_model, nn.DataParallel))
            # But it has 'module' attribute
            self.assertTrue(hasattr(wrapped_model, 'module'))

            # This should work - the function should detect the wrapper via hasattr
            # and add 'module.' prefix when loading
            start_epoch, best_dice = load_checkpoint(save_path, wrapped_model)

            # Verify load was successful
            self.assertEqual(start_epoch, 21)
            self.assertEqual(best_dice, 0.9)
            # The inner model should have the correct weights
            self.assertAlmostEqual(
                wrapped_model.module.conv.weight.sum().item(),
                model.conv.weight.sum().item(),
                places=5
            )

    def test_load_checkpoint_handles_ddp_checkpoint_to_regular_model(self):
        """load_checkpoint() should handle loading DDP checkpoint to regular model"""
        import tempfile
        from utils.checkpoint import save_checkpoint, load_checkpoint

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 2, 1)

            def forward(self, x):
                return self.conv(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint with 'module.' prefix (simulating DDP save)
            model = SimpleModel()
            with torch.no_grad():
                model.conv.weight.fill_(2.0)
                model.conv.bias.fill_(1.0)

            # Simulate DDP state dict (with 'module.' prefix)
            ddp_state_dict = {f'module.{k}': v for k, v in model.state_dict().items()}

            state = {
                'epoch': 5,
                'model_state_dict': ddp_state_dict,
                'best_dice': 0.6,
            }
            save_path = os.path.join(tmpdir, 'ddp_checkpoint.pth')
            save_checkpoint(state, tmpdir, 'ddp_checkpoint.pth')

            # Load into regular model
            model2 = SimpleModel()
            start_epoch, best_dice = load_checkpoint(save_path, model2)

            # Verify the weights were loaded correctly
            self.assertEqual(start_epoch, 6)
            self.assertEqual(best_dice, 0.6)
            self.assertAlmostEqual(model2.conv.weight.sum().item(), model.conv.weight.sum().item(), places=5)


class TestValidateDDPChanges(unittest.TestCase):
    """Test validate() function DDP-related changes"""

    def test_validate_calls_sync_across_processes_when_distributed(self):
        """validate() should call metric.sync_across_processes() when in distributed mode"""
        from train import validate, is_distributed
        from utils.metrics import DiceMetric
        from torch.utils.data import DataLoader, TensorDataset

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 5, 1)

            def forward(self, x):
                return self.conv(x)

        model = SimpleModel()
        device = torch.device("cpu")

        # Create simple dataset and loader
        inputs = torch.randn(4, 1, 4, 4, 4)
        targets = torch.randint(0, 5, (4, 4, 4, 4))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        metric = DiceMetric(num_classes=5)

        # Mock is_distributed to return True, is_main_process to return True, and sync_across_processes
        with mock.patch('train.is_distributed', return_value=True):
            with mock.patch('train.is_main_process', return_value=True):
                with mock.patch.object(metric, 'sync_across_processes') as mock_sync:
                    validate(model, loader, criterion, metric, device)
                    # sync_across_processes should be called
                    mock_sync.assert_called_once()

    def test_validate_does_not_call_sync_when_not_distributed(self):
        """validate() should NOT call metric.sync_across_processes() when not distributed"""
        from train import validate
        from utils.metrics import DiceMetric
        from torch.utils.data import DataLoader, TensorDataset

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 5, 1)

            def forward(self, x):
                return self.conv(x)

        model = SimpleModel()
        device = torch.device("cpu")

        inputs = torch.randn(4, 1, 4, 4, 4)
        targets = torch.randint(0, 5, (4, 4, 4, 4))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=2)

        criterion = nn.CrossEntropyLoss()
        metric = DiceMetric(num_classes=5)

        # is_distributed returns False by default (not mocked)
        with mock.patch.object(metric, 'sync_across_processes') as mock_sync:
            validate(model, loader, criterion, metric, device)
            # sync_across_processes should NOT be called
            mock_sync.assert_not_called()


class TestMainDDPIntegration(unittest.TestCase):
    """Test main() function DDP integration aspects"""

    def test_distributed_sampler_created_when_distributed(self):
        """DistributedSampler should be created when is_distributed() returns True"""
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import DataLoader, TensorDataset

        # Test the pattern that main() should follow
        inputs = torch.randn(8, 1, 4, 4, 4)
        targets = torch.randint(0, 5, (8, 4, 4, 4))
        dataset = TensorDataset(inputs, targets)

        # When distributed - we need to pass num_replicas and rank explicitly
        # since process group is not initialized
        with mock.patch('train.is_distributed', return_value=True):
            from train import is_distributed
            if is_distributed():
                # In real code, we'd use DistributedSampler without explicit args
                # For testing without init_process_group, we pass them explicitly
                train_sampler = DistributedSampler(
                    dataset, shuffle=True, num_replicas=2, rank=0
                )
            else:
                train_sampler = None

            self.assertIsNotNone(train_sampler)
            self.assertIsInstance(train_sampler, DistributedSampler)

    def test_no_sampler_when_not_distributed(self):
        """No sampler should be created when not distributed"""
        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import TensorDataset

        inputs = torch.randn(8, 1, 4, 4, 4)
        targets = torch.randint(0, 5, (8, 4, 4, 4))
        dataset = TensorDataset(inputs, targets)

        # When NOT distributed (default)
        from train import is_distributed
        train_sampler = DistributedSampler(dataset, shuffle=True) if is_distributed() else None

        self.assertIsNone(train_sampler)

    def test_model_state_unwrap_works_for_both_dp_and_ddp(self):
        """Model state unwrapping should work for both DataParallel and DDP"""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv3d(1, 2, 1)

            def forward(self, x):
                return self.conv(x)

        # Test with regular model
        model = SimpleModel()
        model_state = (
            model.module.state_dict()
            if hasattr(model, 'module')
            else model.state_dict()
        )
        self.assertIn('conv.weight', model_state)

        # Test with wrapped model (simulates DDP)
        class Wrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

        wrapped = Wrapper(SimpleModel())
        model_state = (
            wrapped.module.state_dict()
            if hasattr(wrapped, 'module')
            else wrapped.state_dict()
        )
        self.assertIn('conv.weight', model_state)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run_tests():
    """Run all DDP tests"""
    print_section("DDP Helper Functions Tests")

    # Run unittest
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestDDPHelperFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestSetupLoggingWithMainProcess))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainOneEpochWithDistributedSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestDiceMetricSyncAcrossProcesses))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckpointDDPHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateDDPChanges))
    suite.addTests(loader.loadTestsFromTestCase(TestMainDDPIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    if result.wasSuccessful():
        print("  All DDP tests passed!")
    else:
        print(f"  Failed: {len(result.failures)}, Errors: {len(result.errors)}")
    print('='*60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
