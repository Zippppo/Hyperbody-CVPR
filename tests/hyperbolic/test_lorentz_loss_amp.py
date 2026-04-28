import torch
import pytest


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for AMP test')
def test_lorentz_ranking_loss_is_stable_under_cuda_autocast():
    from models.hyperbolic.lorentz_loss import LorentzRankingLoss

    loss_fn = LorentzRankingLoss(
        margin=0.1,
        num_samples_per_class=16,
        num_negatives=4,
        t_start=2.0,
        t_end=0.1,
        warmup_epochs=5,
    ).cuda()
    loss_fn.set_epoch(10)

    # Large magnitudes stress matrix multiplications used by pairwise distances.
    voxel_emb = torch.randn(1, 32, 4, 4, 4, device='cuda') * 120.0
    labels = torch.randint(0, 20, (1, 4, 4, 4), device='cuda')
    label_emb = torch.randn(20, 32, device='cuda') * 120.0

    with torch.autocast(device_type='cuda'):
        loss = loss_fn(voxel_emb, labels, label_emb)

    assert torch.isfinite(loss), 'Loss should remain finite under autocast'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for AMP test')
def test_lorentz_tree_ranking_loss_is_stable_under_cuda_autocast():
    from models.hyperbolic.lorentz_loss import LorentzTreeRankingLoss

    num_classes = 20
    tree_dist = torch.rand(num_classes, num_classes)
    tree_dist = (tree_dist + tree_dist.T) / 2
    tree_dist.fill_diagonal_(0)

    loss_fn = LorentzTreeRankingLoss(
        tree_dist_matrix=tree_dist,
        margin=0.1,
        num_samples_per_class=16,
        num_negatives=4,
        t_start=2.0,
        t_end=0.1,
        warmup_epochs=5,
    ).cuda()
    loss_fn.set_epoch(10)

    voxel_emb = torch.randn(1, 32, 4, 4, 4, device='cuda') * 120.0
    labels = torch.randint(0, num_classes, (1, 4, 4, 4), device='cuda')
    label_emb = torch.randn(num_classes, 32, device='cuda') * 120.0

    with torch.autocast(device_type='cuda'):
        loss = loss_fn(voxel_emb, labels, label_emb)

    assert torch.isfinite(loss), 'Tree loss should remain finite under autocast'
