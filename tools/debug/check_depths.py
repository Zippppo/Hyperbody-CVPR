"""
Quick check of class depth distribution and label embedding initialization.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.organ_hierarchy import load_organ_hierarchy

# Load class info
with open("Dataset/dataset_info.json") as f:
    class_names = json.load(f)["class_names"]

class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)

print("Class Depth Distribution:")
print("="*60)

depths = list(class_depths.values())
min_depth = min(depths)
max_depth = max(depths)

print(f"Min depth: {min_depth}")
print(f"Max depth: {max_depth}")
print(f"Depth range: {max_depth - min_depth}")

# Count classes at each depth
from collections import Counter
depth_counts = Counter(depths)
print(f"\nClasses per depth level:")
for depth in sorted(depth_counts.keys()):
    print(f"  Depth {depth}: {depth_counts[depth]} classes")

# Show how norms would be computed
min_radius = 0.1
max_radius = 2.0
depth_range = max_depth - min_depth if max_depth > min_depth else 1

print(f"\nNorm computation (min_radius={min_radius}, max_radius={max_radius}):")
print(f"  Formula: norm = min_radius + (max_radius - min_radius) * (depth - min_depth) / depth_range")
print(f"  depth_range = {depth_range}")

for depth in sorted(depth_counts.keys()):
    normalized_depth = (depth - min_depth) / depth_range
    norm = min_radius + (max_radius - min_radius) * normalized_depth
    print(f"  Depth {depth}: normalized={normalized_depth:.3f}, norm={norm:.3f}")

# Show first 10 classes with their depths and expected norms
print(f"\nFirst 10 classes:")
for i in range(min(10, len(class_names))):
    depth = class_depths.get(i, min_depth)
    normalized_depth = (depth - min_depth) / depth_range
    norm = min_radius + (max_radius - min_radius) * normalized_depth
    print(f"  Class {i} ({class_names[i]}): depth={depth}, norm={norm:.3f}")
