"""Check symmetry of contact matrix and graph distance matrix."""

import json
import torch

def main():
    contact = torch.load("Dataset/contact_matrix.pt", map_location="cpu")
    graph = torch.load("Dataset/graph_distance_matrix.pt", map_location="cpu")

    with open("Dataset/dataset_info.json") as f:
        class_names = json.load(f)["class_names"]

    for name, mat in [("Contact Matrix", contact), ("Graph Distance Matrix", graph)]:
        diff = (mat - mat.T).abs()
        is_sym = torch.allclose(mat, mat.T)
        asym_count = int((diff > 1e-6).sum().item())
        total = mat.shape[0] * (mat.shape[0] - 1)  # off-diagonal entries

        print(f"=== {name} ({mat.shape[0]}x{mat.shape[1]}) ===")
        print(f"  Symmetric: {is_sym}")
        print(f"  Asymmetric entries: {asym_count}/{total}")
        print(f"  Max |M - M^T|:  {diff.max().item():.6f}")
        print(f"  Mean |M - M^T|: {diff.mean().item():.6f}")

        if asym_count > 0:
            # Show top-5 most asymmetric pairs
            diff.fill_diagonal_(0)
            flat = diff.flatten()
            topk = flat.topk(min(10, asym_count))
            print(f"  Top asymmetric pairs:")
            seen = set()
            for val, idx in zip(topk.values, topk.indices):
                r = idx.item() // mat.shape[0]
                c = idx.item() % mat.shape[0]
                pair = (min(r, c), max(r, c))
                if pair in seen:
                    continue
                seen.add(pair)
                print(
                    f"    [{r:2d}] {class_names[r]:30s} <-> [{c:2d}] {class_names[c]:30s}  "
                    f"M[{r},{c}]={mat[r,c]:.4f}  M[{c},{r}]={mat[c,r]:.4f}  diff={val:.4f}"
                )
                if len(seen) >= 5:
                    break
        print()


if __name__ == "__main__":
    main()
