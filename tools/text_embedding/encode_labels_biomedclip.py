"""
    conda activate bioclip
    python scripts/body/text_embedding/encode_labels_biomedclip.py

"""

import yaml
import torch
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer


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
    Format a label for better encoding by BiomedCLIP.
    Converts underscores to spaces and adds context.

    Example: 'liver' -> 'liver organ'
             'kidney_left' -> 'left kidney organ'
             'visceral_system' -> 'visceral system'
    """
    # Replace underscores with spaces
    formatted = label.replace('_', ' ')

    # Handle left/right naming convention
    if ' left' in formatted:
        formatted = 'left ' + formatted.replace(' left', '')
    elif ' right' in formatted:
        formatted = 'right ' + formatted.replace(' right', '')

    return formatted


def encode_labels_with_biomedclip(
    labels: dict,
    model_name: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    context_length: int = 256,
    use_template: bool = False,
    template: str = 'a medical image of '
) -> dict:
    """
    Encode labels using BiomedCLIP text encoder.

    Args:
        labels: dict of {label_name: label_id}
        model_name: HuggingFace model name
        context_length: Maximum context length for tokenizer
        use_template: Whether to prepend a template to labels
        template: Template string to prepend

    Returns:
        dict containing:
            - 'label_names': list of original label names
            - 'label_ids': tensor of label IDs (or -1 for category labels)
            - 'formatted_labels': list of formatted label strings
            - 'embeddings': tensor of shape (num_labels, embed_dim)
    """
    print(f"Loading BiomedCLIP model from {model_name}...")
    model, preprocess = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    # Prepare label lists
    label_names = list(labels.keys())
    label_ids = [labels[name] if labels[name] is not None else -1 for name in label_names]

    # Format labels for encoding
    formatted_labels = [format_label_for_encoding(name) for name in label_names]

    if use_template:
        text_inputs = [template + label for label in formatted_labels]
    else:
        text_inputs = formatted_labels

    print(f"\nEncoding {len(label_names)} labels...")
    print("Sample formatted labels:")
    for i in range(min(5, len(formatted_labels))):
        print(f"  {label_names[i]} -> '{text_inputs[i]}'")

    # Tokenize and encode
    texts = tokenizer(text_inputs, context_length=context_length).to(device)

    with torch.no_grad():
        # Get text features from model
        # The model's encode_text method returns normalized text features
        text_features = model.encode_text(texts)

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
    # Paths - script is in scripts/body/text_embedding/, so go up 4 levels
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    tree_file = project_root / 'Dataset' / 'label_definitions.yaml'
    output_dir = project_root / 'Dataset' / 'text_embeddings'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'biomedclip_label_embeddings.pt'

    print(f"Project root: {project_root}")
    print(f"Reading label definitions from: {tree_file}")

    # Load labels from YAML
    labels = load_label_definitions(tree_file)
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
    result = encode_labels_with_biomedclip(labels)

    # Save results
    print(f"\nSaving embeddings to: {output_file}")
    torch.save(result, output_file)

    # Also create a version with only labeled classes (for training)
    labeled_result = encode_labels_with_biomedclip(labels_with_id)
    labeled_output = output_dir / 'biomedclip_labeled_embeddings.pt'
    print(f"Saving labeled-only embeddings to: {labeled_output}")
    torch.save(labeled_result, labeled_output)

    print("\nDone!")
    print(f"Total embeddings saved: {result['embeddings'].shape}")
    print(f"Labeled embeddings saved: {labeled_result['embeddings'].shape}")


if __name__ == '__main__':
    main()
