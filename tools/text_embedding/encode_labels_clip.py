"""
    conda activate meru
    python scripts/body/text_embedding/encode_labels_clip.py

"""

import yaml
import torch
import clip
from pathlib import Path


def load_label_definitions(yaml_file: str) -> dict:
    """
    Load label definitions from label_definitions.yaml.

    Returns:
        dict: {label_name: label_id} for labeled classes,
              {label_name: None} for category labels.
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    labels = {}
    for entry in data['labeled']:
        labels[entry['name']] = entry['id']
    for name in data['categories']:
        labels[name] = None

    return labels


def format_label_for_encoding(label: str) -> str:
    """
    Format a label for better encoding by CLIP.
    Converts underscores to spaces and handles left/right conventions.

    Example: 'liver' -> 'liver'
             'kidney_left' -> 'left kidney'
             'visceral_system' -> 'visceral system'
    """
    formatted = label.replace('_', ' ')

    # Handle left/right naming convention
    if ' left' in formatted:
        formatted = 'left ' + formatted.replace(' left', '')
    elif ' right' in formatted:
        formatted = 'right ' + formatted.replace(' right', '')

    return formatted


def encode_labels_with_clip(
    labels: dict,
    model_name: str = "ViT-L/14",
) -> dict:
    """
    Encode labels using CLIP text encoder.

    Args:
        labels: dict of {label_name: label_id}
        model_name: CLIP model name (e.g., "ViT-L/14", "ViT-B/32")

    Returns:
        dict containing:
            - 'label_names': list of original label names
            - 'label_ids': tensor of label IDs (or -1 for category labels)
            - 'formatted_labels': list of formatted label strings
            - 'embeddings': tensor of shape (num_labels, embed_dim)
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading CLIP model: {model_name}...")
    model, _ = clip.load(model_name, device=device)
    model.eval()

    # Prepare label lists
    label_names = list(labels.keys())
    label_ids = [labels[name] if labels[name] is not None else -1 for name in label_names]

    # Format labels for encoding
    formatted_labels = [format_label_for_encoding(name) for name in label_names]

    print(f"\nEncoding {len(label_names)} labels...")
    print("Sample formatted labels:")
    for i in range(min(5, len(formatted_labels))):
        print(f"  {label_names[i]} -> '{formatted_labels[i]}'")

    # Tokenize and encode
    tokens = clip.tokenize(formatted_labels).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        # Normalize to unit vectors
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"Embedding shape: {text_features.shape}")

    return {
        'label_names': label_names,
        'label_ids': torch.tensor(label_ids),
        'formatted_labels': formatted_labels,
        'embeddings': text_features.cpu()
    }


def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    tree_file = project_root / 'Dataset' / 'label_definitions.yaml'
    output_dir = project_root / 'Dataset' / 'text_embeddings'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'clip_label_embeddings.pt'
    model_name = "ViT-L/14"  # Standard CLIP model, will be downloaded if not cached

    print(f"Project root: {project_root}")
    print(f"Reading label definitions from: {tree_file}")

    # Load labels from YAML
    labels = load_label_definitions(str(tree_file))
    print(f"Found {len(labels)} labels")

    # Separate labels with IDs and category labels
    labels_with_id = {k: v for k, v in labels.items() if v is not None}
    category_labels = {k: v for k, v in labels.items() if v is None}

    print(f"\nLabels with IDs ({len(labels_with_id)}):")
    for name, id_ in sorted(labels_with_id.items(), key=lambda x: x[1]):
        print(f"  {id_:3d}: {name}")

    print(f"\nCategory labels ({len(category_labels)}):")
    for name in sorted(category_labels.keys()):
        print(f"  {name}")

    # Encode all labels
    result = encode_labels_with_clip(labels, model_name=model_name)

    # Save results
    print(f"\nSaving embeddings to: {output_file}")
    torch.save(result, output_file)

    # Also create a version with only labeled classes (for training)
    labeled_result = encode_labels_with_clip(labels_with_id, model_name=model_name)
    labeled_output = output_dir / 'clip_labeled_embeddings.pt'
    print(f"Saving labeled-only embeddings to: {labeled_output}")
    torch.save(labeled_result, labeled_output)

    print("\nDone!")
    print(f"Total embeddings saved: {result['embeddings'].shape}")
    print(f"Labeled embeddings saved: {labeled_result['embeddings'].shape}")


if __name__ == '__main__':
    main()
