"""
Lorentz (hyperboloid) model operations for hyperbolic geometry.

The Lorentz model represents hyperbolic space as a hyperboloid in Minkowski space.
We store only spatial components; time component is computed as needed:
    x_time = sqrt(1/curv + ||x_space||^2)

Reference: HyperPath (models/lorentz.py)
"""
import math
import torch
from torch import Tensor


def exp_map0(v: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Exponential map from tangent space at origin to Lorentz manifold.

    Args:
        v: Tangent vectors at origin [..., D]
        curv: Curvature (positive value for negative curvature -curv)
        eps: Small value for numerical stability

    Returns:
        Points on Lorentz manifold (spatial components only) [..., D]
    """
    # ||v|| scaled by sqrt(curv)
    v_norm = torch.norm(v, dim=-1, keepdim=True)
    rc_vnorm = math.sqrt(curv) * v_norm

    # Clamp sinh input to prevent overflow: asinh(2^15) ≈ 11.09
    sinh_input = torch.clamp(rc_vnorm, min=eps, max=math.asinh(2**15))

    # x = sinh(sqrt(c)*||v||) * v / (sqrt(c)*||v||)
    # For numerical stability, handle small norms specially
    scale = torch.sinh(sinh_input) / torch.clamp(rc_vnorm, min=eps)
    return scale * v


def log_map0(x: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Logarithmic map from Lorentz manifold to tangent space at origin.

    Args:
        x: Points on Lorentz manifold (spatial components only) [..., D]
        curv: Curvature (positive value for negative curvature -curv)
        eps: Small value for numerical stability

    Returns:
        Tangent vectors at origin [..., D]
    """
    # Compute time component: x_time = sqrt(1/curv + ||x||^2)
    x_sqnorm = torch.sum(x**2, dim=-1, keepdim=True)
    x_time = torch.sqrt(1.0 / curv + x_sqnorm)

    # Distance from origin: acosh(sqrt(curv) * x_time)
    # Note: sqrt(curv) * x_time >= 1 always (equality at origin)
    acosh_input = math.sqrt(curv) * x_time
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps))

    # v = distance * x / ||x||
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    scale = distance / torch.clamp(x_norm, min=eps)

    # Handle origin case: when x ≈ 0, return 0
    return torch.where(x_norm > eps, scale * x, torch.zeros_like(x))


def pointwise_dist(x: Tensor, y: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Element-wise geodesic distance between corresponding points.

    Args:
        x: Points on Lorentz manifold [..., D]
        y: Points on Lorentz manifold [..., D] (same shape as x)
        curv: Curvature
        eps: Small value for numerical stability

    Returns:
        Geodesic distances [...] (last dimension reduced)
    """
    # Compute time components
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1.0 / curv + torch.sum(y**2, dim=-1))

    # Lorentz inner product: <x,y>_L = x_s · y_s - x_t * y_t
    spatial_inner = torch.sum(x * y, dim=-1)
    lorentz_inner = spatial_inner - x_time * y_time

    # Distance: acosh(-curv * <x,y>_L) / sqrt(curv)
    acosh_input = -curv * lorentz_inner
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps)) / math.sqrt(curv)

    return distance


def pairwise_dist(x: Tensor, y: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    All-pairs geodesic distance between two sets of points.

    Args:
        x: Points on Lorentz manifold [N, D]
        y: Points on Lorentz manifold [M, D]
        curv: Curvature
        eps: Small value for numerical stability

    Returns:
        Distance matrix [N, M]
    """
    # Compute time components
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1, keepdim=True))  # [N, 1]
    y_time = torch.sqrt(1.0 / curv + torch.sum(y**2, dim=-1, keepdim=True))  # [M, 1]

    # Lorentz inner product: <x,y>_L = x_s @ y_s.T - x_t @ y_t.T
    spatial_inner = x @ y.T  # [N, M]
    time_inner = x_time @ y_time.T  # [N, M]
    lorentz_inner = spatial_inner - time_inner

    # Distance: acosh(-curv * <x,y>_L) / sqrt(curv)
    acosh_input = -curv * lorentz_inner
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps)) / math.sqrt(curv)

    return distance


def distance_to_origin(x: Tensor, curv: float = 1.0, eps: float = 1e-7) -> Tensor:
    """
    Geodesic distance from origin for each point.

    Args:
        x: Points on Lorentz manifold [..., D]
        curv: Curvature
        eps: Small value for numerical stability

    Returns:
        Distances from origin [...]
    """
    # Time component of x
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1))

    # Origin time component: sqrt(1/curv)
    origin_time = math.sqrt(1.0 / curv)

    # Lorentz inner product with origin: 0 - x_t * origin_t = -x_t * origin_t
    lorentz_inner = -x_time * origin_time

    # Distance
    acosh_input = -curv * lorentz_inner
    distance = torch.acosh(torch.clamp(acosh_input, min=1.0 + eps)) / math.sqrt(curv)

    return distance


def lorentz_to_poincare(x: Tensor, curv: float = 1.0) -> Tensor:
    """
    Project Lorentz spatial components to Poincare disk.

    Useful for 2D visualization of hyperbolic embeddings.

    Args:
        x: Points on Lorentz manifold (spatial components) [..., D]
        curv: Curvature

    Returns:
        Points in Poincare disk [..., D] (||p|| < 1)
    """
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    return x / (1.0 + x_time)
