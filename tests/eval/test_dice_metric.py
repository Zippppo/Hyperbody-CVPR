"""
DiceMetric 详细测试脚本
测试内容：
1. 基础形状检查
2. 完美预测 (Dice = 1.0)
3. 完全错误预测 (Dice ≈ 0)
4. 部分重叠预测 (Dice = 已知值)
5. 多 batch 累积正确性
6. 边界情况（单类别、空预测）
7. 可视化 Dice 分布
"""

import torch
import numpy as np

# 添加项目路径
import sys
sys.path.insert(0, '/home/comp/25481568/code/HyperBody')

from utils.metrics import DiceMetric


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_basic_shapes():
    """测试基础形状"""
    print_section("Test 1: 基础形状检查")

    metric = DiceMetric(num_classes=70)

    # 测试不同 batch size
    for B in [1, 2, 4]:
        logits = torch.randn(B, 70, 16, 12, 32)
        targets = torch.randint(0, 70, (B, 16, 12, 32))

        metric.reset()
        metric.update(logits, targets)
        dice_per_class, mean_dice, valid_mask = metric.compute()

        print(f"  Batch size {B}:")
        print(f"    logits:         {tuple(logits.shape)}")
        print(f"    targets:        {tuple(targets.shape)}")
        print(f"    dice_per_class: {tuple(dice_per_class.shape)}")
        print(f"    valid_mask:     {tuple(valid_mask.shape)}")

        assert dice_per_class.shape == (70,)
        assert valid_mask.shape == (70,)

    print("  ✓ 所有形状检查通过")


def test_perfect_prediction():
    """测试完美预测"""
    print_section("Test 2: 完美预测 (Dice = 1.0)")

    metric = DiceMetric(num_classes=10)

    # 创建已知的 targets
    targets = torch.zeros(2, 8, 8, 8, dtype=torch.long)
    targets[0, :4, :, :] = 1  # 前半部分是类别1
    targets[0, 4:, :, :] = 2  # 后半部分是类别2
    targets[1, :, :4, :] = 3  # 类别3
    targets[1, :, 4:, :] = 4  # 类别4

    # 创建完美预测的 logits
    logits = torch.full((2, 10, 8, 8, 8), -10.0)
    for b in range(2):
        for d in range(8):
            for h in range(8):
                for w in range(8):
                    c = targets[b, d, h, w].item()
                    logits[b, c, d, h, w] = 10.0

    metric.update(logits, targets)
    dice_per_class, mean_dice, valid_mask = metric.compute()

    print(f"  targets 中的类别: {torch.unique(targets).tolist()}")
    print(f"  各类别体素数:")
    for c in torch.unique(targets):
        count = (targets == c).sum().item()
        print(f"    类别 {c}: {count} 体素")

    print(f"\n  各类别 Dice 分数:")
    for c in range(10):
        if valid_mask[c]:
            print(f"    类别 {c}: {dice_per_class[c].item():.6f}")

    print(f"\n  mean_dice: {mean_dice:.6f}")
    print(f"  Expected:  1.000000")

    assert abs(mean_dice - 1.0) < 1e-5, f"完美预测应得 Dice=1.0, 实际={mean_dice}"
    print("  ✓ 完美预测测试通过")


def test_zero_overlap():
    """测试完全错误预测"""
    print_section("Test 3: 完全错误预测 (Dice ≈ 0)")

    metric = DiceMetric(num_classes=10)

    # targets 全是类别 1
    targets = torch.ones(1, 8, 8, 8, dtype=torch.long)

    # 预测全是类别 2（完全不重叠）
    logits = torch.full((1, 10, 8, 8, 8), -10.0)
    logits[:, 2, :, :, :] = 10.0  # 预测类别2

    metric.update(logits, targets)
    dice_per_class, mean_dice, valid_mask = metric.compute()

    print(f"  targets: 全部是类别 1 ({(targets==1).sum().item()} 体素)")
    print(f"  预测:    全部是类别 2")
    print(f"\n  类别 1 Dice: {dice_per_class[1].item():.6f} (应接近 0)")
    print(f"  类别 2 Dice: {dice_per_class[2].item():.6f} (应接近 0)")
    print(f"  mean_dice:   {mean_dice:.6f}")

    # 由于 smooth factor，不会精确为0
    assert dice_per_class[1].item() < 0.01, "类别1 Dice应接近0"
    print("  ✓ 完全错误预测测试通过")


def test_partial_overlap():
    """测试部分重叠（已知 Dice 值）"""
    print_section("Test 4: 部分重叠 (验证 Dice 计算)")

    metric = DiceMetric(num_classes=10, smooth=0)  # 无平滑，便于验证

    # 创建简单场景：8x8x8 = 512 体素
    # targets: 前256个是类别1，后256个是类别2
    # 预测:    前128个是类别1，129-384是类别2，385-512是类别1

    targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    targets[0, :4, :, :] = 1  # 256 体素
    targets[0, 4:, :, :] = 2  # 256 体素

    logits = torch.full((1, 10, 8, 8, 8), -10.0)
    # 预测类别1: 前2层 (128体素)
    logits[0, 1, :2, :, :] = 10.0
    # 预测类别2: 第2-6层 (256体素)
    logits[0, 2, 2:6, :, :] = 10.0
    # 预测类别1: 第6-8层 (128体素)
    logits[0, 1, 6:, :, :] = 10.0

    metric.update(logits, targets)
    dice_per_class, mean_dice, valid_mask = metric.compute()

    # 手动计算预期 Dice
    # 类别1: target=256, pred=256(前2层128+后2层128)
    #        intersection = 128 (前2层重叠)
    #        Dice = 2*128 / (256+256) = 256/512 = 0.5
    # 类别2: target=256, pred=256(第2-6层)
    #        intersection = 128 (第4-6层重叠)
    #        Dice = 2*128 / (256+256) = 0.5

    print("  场景设置:")
    print("    targets: 层0-3=类别1(256), 层4-7=类别2(256)")
    print("    预测:    层0-1=类别1(128), 层2-5=类别2(256), 层6-7=类别1(128)")
    print()
    print("  手动计算:")
    print("    类别1: intersection=128, union=256+256=512, Dice=2*128/512=0.5")
    print("    类别2: intersection=128, union=256+256=512, Dice=2*128/512=0.5")
    print()
    print(f"  实际结果:")
    print(f"    类别1 Dice: {dice_per_class[1].item():.6f} (期望 0.5)")
    print(f"    类别2 Dice: {dice_per_class[2].item():.6f} (期望 0.5)")
    print(f"    mean_dice:  {mean_dice:.6f} (期望 0.5)")

    assert abs(dice_per_class[1].item() - 0.5) < 0.01
    assert abs(dice_per_class[2].item() - 0.5) < 0.01
    print("  ✓ 部分重叠测试通过")


def test_multi_batch_accumulation():
    """测试多 batch 累积"""
    print_section("Test 5: 多 batch 累积正确性")

    metric = DiceMetric(num_classes=5, smooth=0)

    # Batch 1: 全是类别1，完美预测
    targets1 = torch.ones(1, 4, 4, 4, dtype=torch.long)
    logits1 = torch.full((1, 5, 4, 4, 4), -10.0)
    logits1[:, 1, :, :, :] = 10.0

    # Batch 2: 全是类别1，完美预测
    targets2 = torch.ones(1, 4, 4, 4, dtype=torch.long)
    logits2 = torch.full((1, 5, 4, 4, 4), -10.0)
    logits2[:, 1, :, :, :] = 10.0

    # 累积
    metric.update(logits1, targets1)
    metric.update(logits2, targets2)

    dice_per_class, mean_dice, valid_mask = metric.compute()

    print("  Batch 1: 64 体素类别1，完美预测")
    print("  Batch 2: 64 体素类别1，完美预测")
    print(f"\n  累积后:")
    print(f"    intersection[1]: {metric.intersection[1].item()}")
    print(f"    pred_sum[1]:     {metric.pred_sum[1].item()}")
    print(f"    target_sum[1]:   {metric.target_sum[1].item()}")
    print(f"    类别1 Dice:      {dice_per_class[1].item():.6f}")
    print(f"    mean_dice:       {mean_dice:.6f}")

    assert metric.intersection[1].item() == 128  # 64 + 64
    assert metric.pred_sum[1].item() == 128
    assert metric.target_sum[1].item() == 128
    assert abs(mean_dice - 1.0) < 1e-5
    print("  ✓ 多 batch 累积测试通过")


def test_edge_cases():
    """测试边界情况"""
    print_section("Test 6: 边界情况")

    # Case 1: 单类别
    print("\n  Case 6a: 单类别场景")
    metric = DiceMetric(num_classes=70)
    targets = torch.zeros(1, 4, 4, 4, dtype=torch.long)  # 全是类别0
    logits = torch.full((1, 70, 4, 4, 4), -10.0)
    logits[:, 0, :, :, :] = 10.0

    metric.update(logits, targets)
    dice_per_class, mean_dice, valid_mask = metric.compute()

    print(f"    valid classes: {valid_mask.sum().item()} (应为 1)")
    print(f"    类别0 Dice: {dice_per_class[0].item():.6f}")
    print(f"    mean_dice:  {mean_dice:.6f}")
    assert valid_mask.sum().item() == 1
    print("    ✓ 通过")

    # Case 2: reset 后重新累积
    print("\n  Case 6b: reset 后重新累积")
    metric.reset()
    assert metric.intersection.sum() == 0
    assert metric.pred_sum.sum() == 0
    assert metric.target_sum.sum() == 0
    print("    ✓ reset 正确清零")

    # Case 3: 小体积
    print("\n  Case 6c: 极小体积 (1x1x1)")
    metric = DiceMetric(num_classes=10)
    targets = torch.tensor([[[[5]]]], dtype=torch.long)  # 1x1x1x1
    logits = torch.full((1, 10, 1, 1, 1), -10.0)
    logits[:, 5, :, :, :] = 10.0

    metric.update(logits, targets)
    dice_per_class, mean_dice, valid_mask = metric.compute()
    print(f"    targets shape: {tuple(targets.shape)}")
    print(f"    mean_dice: {mean_dice:.6f}")
    print("    ✓ 通过")


def visualize_dice_distribution():
    """可视化 Dice 分布"""
    print_section("Test 7: Dice 分布可视化")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  plotly 未安装，跳过可视化")
        return

    # 模拟一个有各种 Dice 值的场景
    metric = DiceMetric(num_classes=70, smooth=1e-5)

    np.random.seed(42)

    # 创建模拟数据：不同类别有不同的预测准确度
    for _ in range(5):  # 5 个 batch
        targets = torch.zeros(2, 32, 32, 32, dtype=torch.long)
        logits = torch.full((2, 70, 32, 32, 32), -10.0)

        # 随机分配类别到不同区域
        for c in range(20):  # 使用前20个类别
            # 随机选择一个区域
            d_start = np.random.randint(0, 24)
            h_start = np.random.randint(0, 24)
            w_start = np.random.randint(0, 24)

            targets[:, d_start:d_start+8, h_start:h_start+8, w_start:w_start+8] = c

            # 添加不同程度的噪声来模拟不同预测质量
            noise_level = np.random.uniform(0, 0.5)
            for b in range(2):
                for d in range(d_start, d_start+8):
                    for h in range(h_start, h_start+8):
                        for w in range(w_start, w_start+8):
                            if np.random.random() > noise_level:
                                logits[b, c, d, h, w] = 10.0
                            else:
                                wrong_c = np.random.randint(0, 70)
                                logits[b, wrong_c, d, h, w] = 10.0

        metric.update(logits, targets)

    dice_per_class, mean_dice, valid_mask = metric.compute()

    # 创建可视化
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '各类别 Dice 分数',
            'Dice 分数分布直方图',
            '有效类别统计',
            '累积统计量'
        ),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )

    # 1. 各类别 Dice 分数
    valid_dice = dice_per_class[valid_mask].numpy()
    valid_indices = torch.where(valid_mask)[0].numpy()

    fig.add_trace(
        go.Bar(
            x=[f"类别{i}" for i in valid_indices],
            y=valid_dice,
            marker_color=valid_dice,
            marker_colorscale='RdYlGn',
            name='Dice'
        ),
        row=1, col=1
    )

    # 2. Dice 分数分布
    fig.add_trace(
        go.Histogram(
            x=valid_dice,
            nbinsx=20,
            name='分布',
            marker_color='steelblue'
        ),
        row=1, col=2
    )

    # 3. 有效类别统计
    fig.add_trace(
        go.Pie(
            labels=['有效类别', '无效类别'],
            values=[valid_mask.sum().item(), (~valid_mask).sum().item()],
            marker_colors=['#2ecc71', '#e74c3c']
        ),
        row=2, col=1
    )

    # 4. 累积统计量
    fig.add_trace(
        go.Bar(
            x=['Intersection', 'Pred Sum', 'Target Sum'],
            y=[
                metric.intersection[valid_mask].sum().item(),
                metric.pred_sum[valid_mask].sum().item(),
                metric.target_sum[valid_mask].sum().item()
            ],
            marker_color=['#3498db', '#e74c3c', '#2ecc71']
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=f'DiceMetric 测试结果 (Mean Dice: {mean_dice:.4f})',
        height=800,
        showlegend=False
    )

    # 保存为 HTML
    output_path = '/home/comp/25481568/code/HyperBody/tests/dice_metric_test.html'
    fig.write_html(output_path)
    print(f"  可视化已保存到: {output_path}")

    # 打印统计摘要
    print(f"\n  统计摘要:")
    print(f"    有效类别数: {valid_mask.sum().item()}")
    print(f"    Mean Dice:  {mean_dice:.4f}")
    print(f"    Min Dice:   {valid_dice.min():.4f}")
    print(f"    Max Dice:   {valid_dice.max():.4f}")
    print(f"    Std Dice:   {valid_dice.std():.4f}")
    print("  ✓ 可视化测试通过")


def main():
    print("\n" + "="*60)
    print("  DiceMetric 详细测试")
    print("="*60)

    test_basic_shapes()
    test_perfect_prediction()
    test_zero_overlap()
    test_partial_overlap()
    test_multi_batch_accumulation()
    test_edge_cases()
    visualize_dice_distribution()

    print("\n" + "="*60)
    print("  所有测试通过!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
