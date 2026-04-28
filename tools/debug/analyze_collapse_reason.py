"""
分析为什么embeddings会向圆心坍缩，而不是在远离圆心的位置互相靠近。

关键问题：
- 初始时 voxel embeddings 远离圆心 (距离 ~1.2-1.5)
- 初始时 label embeddings 也远离圆心 (距离 ~0.1-2.0)
- 它们需要互相靠近，为什么会一起向圆心移动？
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.hyperbolic.lorentz_ops import exp_map0, pointwise_dist, distance_to_origin


def analyze_gradient_direction():
    """
    分析在双曲空间中，triplet loss的梯度方向。

    关键洞察：在双曲空间中，"靠近"的最短路径是什么？
    """
    print("="*70)
    print("双曲空间中的梯度分析")
    print("="*70)

    curv = 1.0

    # 场景1: 两个点都在远离圆心的位置，但方向不同
    print("\n场景1: 两个点在不同方向，都远离圆心")
    print("-"*50)

    # 创建两个tangent vectors，方向不同
    tangent_a = torch.tensor([1.5, 0.0], requires_grad=True)  # 沿x轴
    tangent_b = torch.tensor([0.0, 1.5], requires_grad=True)  # 沿y轴

    # 映射到双曲空间
    point_a = exp_map0(tangent_a, curv)
    point_b = exp_map0(tangent_b, curv)

    # 计算距离
    dist = pointwise_dist(point_a.unsqueeze(0), point_b.unsqueeze(0), curv)

    print(f"  Point A (tangent): {tangent_a.detach().numpy()}")
    print(f"  Point B (tangent): {tangent_b.detach().numpy()}")
    print(f"  Point A 到圆心距离: {distance_to_origin(point_a.unsqueeze(0), curv).item():.4f}")
    print(f"  Point B 到圆心距离: {distance_to_origin(point_b.unsqueeze(0), curv).item():.4f}")
    print(f"  A-B 之间距离: {dist.item():.4f}")

    # 计算梯度 - 最小化距离
    dist.backward()

    print(f"\n  梯度 (最小化距离的方向):")
    print(f"    ∂dist/∂tangent_a = {tangent_a.grad.numpy()}")
    print(f"    ∂dist/∂tangent_b = {tangent_b.grad.numpy()}")

    # 分析梯度方向
    grad_a_norm = tangent_a.grad.norm().item()
    grad_a_radial = (tangent_a.grad * tangent_a.detach()).sum().item() / (tangent_a.detach().norm().item() + 1e-8)

    print(f"\n  梯度分析 (Point A):")
    print(f"    梯度模长: {grad_a_norm:.4f}")
    print(f"    径向分量 (负=向圆心): {grad_a_radial:.4f}")

    if grad_a_radial < 0:
        print("    → 梯度指向圆心方向!")
    else:
        print("    → 梯度指向远离圆心方向")


def analyze_why_collapse_to_origin():
    """
    深入分析为什么会向圆心坍缩。
    """
    print("\n" + "="*70)
    print("为什么会向圆心坍缩？")
    print("="*70)

    curv = 1.0

    # 模拟训练场景：多个voxel embeddings，一个label embedding
    print("\n场景: 多个voxels需要靠近同一个label")
    print("-"*50)

    # Label embedding (可学习)
    label_tangent = torch.tensor([1.0, 0.0], requires_grad=True)

    # 多个voxel embeddings，分布在不同方向 (假设这些是固定的，来自网络输出)
    voxel_tangents = torch.tensor([
        [0.8, 0.3],
        [0.9, -0.2],
        [1.1, 0.1],
        [0.7, -0.4],
        [1.2, 0.5],
    ], requires_grad=False)

    # 映射到双曲空间
    label_point = exp_map0(label_tangent, curv)
    voxel_points = exp_map0(voxel_tangents, curv)

    print(f"  Label tangent: {label_tangent.detach().numpy()}")
    print(f"  Label 到圆心距离: {distance_to_origin(label_point.unsqueeze(0), curv).item():.4f}")
    print(f"  Voxels 到圆心距离: {distance_to_origin(voxel_points, curv).mean().item():.4f}")

    # 计算所有voxel到label的距离之和
    total_dist = 0
    for i in range(len(voxel_tangents)):
        d = pointwise_dist(voxel_points[i:i+1], label_point.unsqueeze(0), curv)
        total_dist = total_dist + d

    mean_dist = total_dist / len(voxel_tangents)
    print(f"  平均 voxel-label 距离: {mean_dist.item():.4f}")

    # 计算梯度
    mean_dist.backward()

    print(f"\n  Label tangent 的梯度: {label_tangent.grad.numpy()}")

    # 分析梯度的径向分量
    grad_radial = (label_tangent.grad * label_tangent.detach()).sum().item()
    grad_radial /= (label_tangent.detach().norm().item() + 1e-8)

    print(f"  梯度径向分量: {grad_radial:.4f}")
    if grad_radial < 0:
        print("  → Label被推向圆心!")

    # 关键洞察
    print("\n" + "="*70)
    print("关键洞察")
    print("="*70)
    print("""
    在双曲空间中，当一个点需要同时靠近多个分散的点时：

    1. 如果这些点分布在不同方向，那么"最优"位置是它们的"中心"
    2. 在双曲空间中，多个点的"中心"往往更靠近圆心
    3. 这是因为双曲空间的几何特性：越靠近边缘，距离增长越快

    类比：想象在一个圆盘上，你要找一个点使得到边缘多个点的距离之和最小
    → 这个点会在圆盘中心附近，而不是边缘

    在我们的场景中：
    - Voxel embeddings 分布在不同方向（因为它们来自不同的空间位置）
    - Label embedding 需要靠近所有属于该类的voxels
    - 最小化距离的最优解是：label移向这些voxels的"重心"
    - 由于voxels分散，这个重心更靠近圆心
    """)


def demonstrate_hyperbolic_centroid():
    """
    演示双曲空间中"重心"的位置。
    """
    print("\n" + "="*70)
    print("双曲空间中的重心位置")
    print("="*70)

    curv = 1.0

    # 在圆周上均匀分布的点
    n_points = 8
    radius = 1.5
    angles = torch.linspace(0, 2*np.pi, n_points+1)[:-1]

    tangents = torch.stack([
        radius * torch.cos(angles),
        radius * torch.sin(angles)
    ], dim=1)

    points = exp_map0(tangents, curv)

    print(f"  {n_points}个点均匀分布在半径={radius}的圆上")
    print(f"  每个点到圆心的双曲距离: {distance_to_origin(points, curv).mean().item():.4f}")

    # 找到使得到所有点距离之和最小的位置
    # 通过梯度下降
    center_tangent = torch.tensor([0.5, 0.5], requires_grad=True)
    optimizer = torch.optim.SGD([center_tangent], lr=0.1)

    print(f"\n  寻找最优中心位置...")
    for step in range(100):
        optimizer.zero_grad()
        center_point = exp_map0(center_tangent, curv)

        total_dist = 0
        for i in range(n_points):
            d = pointwise_dist(center_point.unsqueeze(0), points[i:i+1], curv)
            total_dist = total_dist + d

        total_dist.backward()
        optimizer.step()

        if step % 20 == 0:
            center_dist = distance_to_origin(center_point.unsqueeze(0), curv).item()
            print(f"    Step {step}: center到圆心距离 = {center_dist:.4f}, 总距离 = {total_dist.item():.4f}")

    final_center = exp_map0(center_tangent.detach(), curv)
    final_dist = distance_to_origin(final_center.unsqueeze(0), curv).item()

    print(f"\n  最终中心位置到圆心距离: {final_dist:.4f}")
    print(f"  原始点到圆心距离: {distance_to_origin(points, curv).mean().item():.4f}")
    print(f"\n  结论: 最优中心比原始点更靠近圆心 ({final_dist:.4f} < {radius:.4f})")


def analyze_voxel_distribution():
    """
    分析实际训练中voxel embeddings的分布。
    """
    print("\n" + "="*70)
    print("Voxel Embeddings 的分布特性")
    print("="*70)
    print("""
    在实际训练中，同一类别的voxels来自：
    - 不同的空间位置 (x, y, z)
    - 不同的batch samples
    - 不同的局部上下文

    这导致同一类别的voxel embeddings在方向上是分散的。

    当label embedding需要靠近所有这些分散的voxels时：
    → 它会被拉向这些voxels的"重心"
    → 这个重心更靠近圆心

    同时，voxel embeddings也在被训练：
    - 它们需要靠近自己的label
    - 但也受到segmentation loss的影响
    - segmentation loss不关心双曲结构

    最终结果：
    - Label embeddings 向圆心移动
    - Voxel embeddings 的分布没有被约束
    - 整个系统失去了层次结构
    """)


if __name__ == "__main__":
    analyze_gradient_direction()
    analyze_why_collapse_to_origin()
    demonstrate_hyperbolic_centroid()
    analyze_voxel_distribution()
