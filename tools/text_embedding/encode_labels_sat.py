"""

    conda activate pasco
    python tools/text_embedding/encode_labels_sat.py

"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer


def get_model_path(model_name: str) -> str:
    """
    Get model path, handling cached models when network is unavailable.
    """
    import os
    from huggingface_hub import snapshot_download

    # Try to get from cache first
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    cached_path = os.path.join(
        cache_dir,
        f"models--{model_name.replace('/', '--')}"
    )

    if os.path.exists(cached_path):
        snapshots = os.path.join(cached_path, 'snapshots')
        if os.path.exists(snapshots):
            versions = os.listdir(snapshots)
            if versions:
                return os.path.join(snapshots, versions[0])

    # Download if not cached
    return snapshot_download(repo_id=model_name)


class TextTower(nn.Module):
    """
    Standalone Text Tower using BioLORD-2023-C.
    Simplified version of Embedding/SAT/model/text_tower.py without DDP.
    """
    def __init__(self, model_name: str = 'FremyCompany/BioLORD-2023-C', max_length: int = 256):
        super().__init__()
        model_path = get_model_path(model_name)
        self.biolord = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = max_length

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling over token embeddings with attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, texts: list) -> torch.Tensor:
        """
        Encode a list of text strings.

        Args:
            texts: List of strings to encode

        Returns:
            Tensor of shape (len(texts), 768)
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        device = next(self.biolord.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.biolord(**encoded)
            pooled = self.mean_pooling(output, encoded['attention_mask'])

        return pooled


def load_label_definitions(json_file: str) -> dict:
    """
    Load label definitions from dataset_info.json.

    Returns:
        dict: {label_name: label_id}
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    labels = {}
    for idx, name in enumerate(data['class_names']):
        labels[name] = idx

    return labels


def format_label_for_encoding(label: str) -> str:
    """
    Format a label for better encoding by BioLORD.
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


def encode_labels_with_biolord(
    labels: dict,
    model_name: str = 'FremyCompany/BioLORD-2023-C',
    normalize: bool = True
) -> dict:
    """
    Encode labels using SAT's BioLORD text encoder.

    Args:
        labels: dict of {label_name: label_id}
        model_name: HuggingFace model name for BioLORD
        normalize: Whether to L2-normalize embeddings

    Returns:
        dict containing:
            - 'label_names': list of original label names
            - 'label_ids': tensor of label IDs
            - 'formatted_labels': list of formatted label strings
            - 'embeddings': tensor of shape (num_labels, 768)
    """
    print(f"Loading BioLORD model from {model_name}...")
    model = TextTower(model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model.to(device)
    model.eval()

    # Prepare label lists
    label_names = list(labels.keys())
    label_ids = [labels[name] for name in label_names]

    # Format labels for encoding
    formatted_labels = [format_label_for_encoding(name) for name in label_names]

    print(f"\nEncoding {len(label_names)} labels...")
    print("Sample formatted labels:")
    for i in range(min(5, len(formatted_labels))):
        print(f"  {label_names[i]} -> '{formatted_labels[i]}'")

    # Encode in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(formatted_labels), batch_size):
        batch = formatted_labels[i:i + batch_size]
        with torch.no_grad():
            embeddings = model(batch)
        all_embeddings.append(embeddings)

    text_features = torch.cat(all_embeddings, dim=0)

    if normalize:
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
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_info_file = project_root / 'Dataset' / 'dataset_info.json'
    output_dir = project_root / 'Dataset' / 'text_embeddings'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'sat_label_embeddings.pt'

    print(f"Reading label definitions from: {dataset_info_file}")

    # Load labels from JSON
    labels = load_label_definitions(dataset_info_file)
    print(f"Found {len(labels)} labels")

    print(f"\nLabels ({len(labels)}):")
    for name, id_ in sorted(labels.items(), key=lambda x: x[1]):
        print(f"  {id_:3d}: {name}")

    # Encode all labels
    result = encode_labels_with_biolord(labels)

    # Save results
    print(f"\nSaving embeddings to: {output_file}")
    torch.save(result, output_file)

    print("\nDone!")
    print(f"Total embeddings saved: {result['embeddings'].shape}")


if __name__ == '__main__':
    main()
