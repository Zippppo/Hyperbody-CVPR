"""
Visual test for EmbeddingTracker.

Generates visualization outputs in docs/visualizations/embedding_tracker_test/
to verify the tracker works correctly with simulated training.
"""
import json
import os
import sys

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.hyperbolic.label_embedding import LorentzLabelEmbedding
from models.hyperbolic.embedding_tracker import EmbeddingTracker, SYSTEM_COLORS
from data.organ_hierarchy import load_organ_hierarchy, load_class_to_system


def run_visual_test():
    """
    Run a visual test simulating 5 epochs of training.

    Outputs:
    - docs/visualizations/embedding_tracker_test/
        - embedding_history.json
        - epoch_000.png (or .html)
        - epoch_001.png (or .html)
        - ...
        - animation.html
    """
    print("=" * 60)
    print("EmbeddingTracker Visual Test")
    print("=" * 60)

    # Load class names and hierarchy
    with open("Dataset/dataset_info.json") as f:
        class_names = json.load(f)["class_names"]

    class_depths = load_organ_hierarchy("Dataset/tree.json", class_names)
    class_to_system = load_class_to_system("Dataset/tree.json", class_names)

    print(f"\n[1] Loaded {len(class_names)} classes")
    print(f"    Depth range: {min(class_depths.values())} - {max(class_depths.values())}")

    # Print system distribution
    system_counts = {}
    for idx, system in class_to_system.items():
        system_counts[system] = system_counts.get(system, 0) + 1
    print(f"\n[2] Organ system distribution:")
    for system, count in sorted(system_counts.items(), key=lambda x: -x[1]):
        color = SYSTEM_COLORS.get(system, "#000000")
        print(f"    {system:15s}: {count:2d} classes  (color: {color})")

    # Create label embedding
    torch.manual_seed(42)
    label_embedding = LorentzLabelEmbedding(
        num_classes=70,
        embed_dim=32,
        class_depths=class_depths,
        min_radius=0.1,
        max_radius=2.0
    )

    print(f"\n[3] Created LorentzLabelEmbedding")
    print(f"    Shape: {label_embedding.tangent_embeddings.shape}")

    # Create tracker
    output_dir = "docs/visualizations"
    model_name = "embedding_tracker_test"
    tracker = EmbeddingTracker(
        model_name=model_name,
        class_names=class_names,
        class_to_system=class_to_system,
        output_dir=output_dir
    )

    print(f"\n[4] Created EmbeddingTracker")
    print(f"    Output directory: {tracker.output_dir}")

    # Simulate training for 5 epochs
    num_epochs = 5
    print(f"\n[5] Simulating {num_epochs} epochs of training...")

    for epoch in range(num_epochs):
        # Record embeddings
        tracker.on_epoch_end(epoch=epoch, label_embedding=label_embedding)

        # Get current stats
        with torch.no_grad():
            tangent_norms = torch.norm(label_embedding.tangent_embeddings, dim=-1)
            mean_norm = tangent_norms.mean().item()
            std_norm = tangent_norms.std().item()

        print(f"    Epoch {epoch}: tangent norm = {mean_norm:.4f} +/- {std_norm:.4f}")

        # Simulate training update (add noise to tangent vectors)
        if epoch < num_epochs - 1:
            with torch.no_grad():
                # Simulate gradient update: move embeddings slightly
                noise = torch.randn_like(label_embedding.tangent_embeddings) * 0.1
                label_embedding.tangent_embeddings.data += noise

    # Verify outputs
    print(f"\n[6] Verifying outputs...")

    output_path = tracker.output_dir
    json_path = os.path.join(output_path, "embedding_history.json")
    animation_path = os.path.join(output_path, "animation.html")

    # Check JSON
    with open(json_path) as f:
        data = json.load(f)

    print(f"    JSON file: {json_path}")
    print(f"      - metadata keys: {list(data['metadata'].keys())}")
    print(f"      - num epochs recorded: {len(data['epochs'])}")
    print(f"      - PCA components shape: {len(data['metadata']['pca_components'])}x{len(data['metadata']['pca_components'][0])}")

    # Check for NaN
    nan_epochs = [e['epoch'] for e in data['epochs'] if e['has_nan']]
    if nan_epochs:
        print(f"      - WARNING: NaN detected in epochs: {nan_epochs}")
    else:
        print(f"      - No NaN detected in any epoch")

    # Check visualization files
    print(f"\n    Visualization files:")
    for epoch in range(num_epochs):
        png_path = os.path.join(output_path, f"epoch_{epoch:03d}.png")
        html_path = os.path.join(output_path, f"epoch_{epoch:03d}.html")
        if os.path.exists(png_path):
            size_kb = os.path.getsize(png_path) / 1024
            print(f"      - epoch_{epoch:03d}.png ({size_kb:.1f} KB)")
        elif os.path.exists(html_path):
            size_kb = os.path.getsize(html_path) / 1024
            print(f"      - epoch_{epoch:03d}.html ({size_kb:.1f} KB)")
        else:
            print(f"      - epoch_{epoch:03d}: MISSING!")

    # Check animation
    if os.path.exists(animation_path):
        size_kb = os.path.getsize(animation_path) / 1024
        print(f"      - animation.html ({size_kb:.1f} KB)")
    else:
        print(f"      - animation.html: MISSING!")

    # Analyze embedding evolution
    print(f"\n[7] Embedding evolution analysis:")

    epochs_data = data['epochs']

    # Compute movement between epochs
    for i in range(1, len(epochs_data)):
        prev_pos = np.array(epochs_data[i-1]['poincare_positions'])
        curr_pos = np.array(epochs_data[i]['poincare_positions'])
        movement = np.linalg.norm(curr_pos - prev_pos, axis=1)
        print(f"    Epoch {i-1} -> {i}: mean movement = {movement.mean():.4f}, max = {movement.max():.4f}")

    # Check Poincare ball constraint (all points should be inside unit ball)
    print(f"\n[8] Poincare ball constraint check:")
    for epoch_data in epochs_data:
        poincare_pos = np.array(epoch_data['poincare_positions'])
        norms = np.linalg.norm(poincare_pos, axis=1)
        max_norm = norms.max()
        if max_norm >= 1.0:
            print(f"    Epoch {epoch_data['epoch']}: WARNING - max norm = {max_norm:.4f} >= 1.0")
        else:
            print(f"    Epoch {epoch_data['epoch']}: OK - max norm = {max_norm:.4f} < 1.0")

    print("\n" + "=" * 60)
    print("Visual test completed!")
    print(f"Open {animation_path} in a browser to view the animation.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = run_visual_test()
    sys.exit(0 if success else 1)
