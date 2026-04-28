import torch
import pytest
import json
import os


class TestVisualization:
    """Test hyperbolic embedding visualization."""

    @pytest.fixture
    def class_depths(self):
        from data.organ_hierarchy import load_organ_hierarchy
        with open("Dataset/dataset_info.json") as f:
            class_names = json.load(f)["class_names"]
        return load_organ_hierarchy("Dataset/tree.json", class_names)

    @pytest.fixture
    def class_names(self):
        with open("Dataset/dataset_info.json") as f:
            return json.load(f)["class_names"]

    @pytest.fixture
    def output_dir(self):
        path = "docs/visualizations/hyperbolic"
        os.makedirs(path, exist_ok=True)
        return path

    def test_poincare_disk_visualization(self, class_depths, class_names, output_dir):
        """Visualize label embeddings in Poincare disk."""
        import plotly.express as px
        import pandas as pd
        from sklearn.decomposition import PCA
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import lorentz_to_poincare, distance_to_origin

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        label_emb = emb()  # [70, 32]

        # Project to Poincare disk
        poincare_emb = lorentz_to_poincare(label_emb)  # [70, 32]

        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(poincare_emb.detach().numpy())

        # Get distances for coloring
        distances = distance_to_origin(label_emb).detach().numpy()

        # Create dataframe
        df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "class_name": class_names,
            "depth": [class_depths[i] for i in range(70)],
            "distance": distances,
        })

        # Create plot
        fig = px.scatter(
            df, x="x", y="y",
            color="depth",
            hover_data=["class_name", "distance"],
            title="Label Embeddings in Poincare Disk (PCA 2D)",
            color_continuous_scale="Viridis"
        )

        output_path = os.path.join(output_dir, "label_emb_poincare.html")
        fig.write_html(output_path)
        assert os.path.exists(output_path)

    def test_distance_matrix_heatmap(self, class_depths, class_names, output_dir):
        """Visualize pairwise distance matrix."""
        import plotly.express as px
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import pairwise_dist

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        label_emb = emb()  # [70, 32]

        # Compute pairwise distances
        dist_matrix = pairwise_dist(label_emb, label_emb).detach().numpy()

        # Create heatmap
        fig = px.imshow(
            dist_matrix,
            x=class_names,
            y=class_names,
            color_continuous_scale="Blues",
            title="Pairwise Lorentz Distances Between Classes"
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            width=1200,
            height=1000
        )

        output_path = os.path.join(output_dir, "class_distance_matrix.html")
        fig.write_html(output_path)
        assert os.path.exists(output_path)

    def test_tsne_visualization(self, class_depths, class_names, output_dir):
        """Visualize label embeddings using t-SNE."""
        import plotly.express as px
        import pandas as pd
        from sklearn.manifold import TSNE
        from models.hyperbolic.label_embedding import LorentzLabelEmbedding
        from models.hyperbolic.lorentz_ops import distance_to_origin

        torch.manual_seed(42)
        emb = LorentzLabelEmbedding(
            num_classes=70,
            embed_dim=32,
            class_depths=class_depths
        )
        label_emb = emb().detach().numpy()  # [70, 32]

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        coords_2d = tsne.fit_transform(label_emb)

        # Get organ system from class name (simplified)
        def get_system(name):
            if "rib" in name or "spine" in name or "skull" in name or "sternum" in name or \
               "scapula" in name or "clavicula" in name or "humerus" in name or \
               "hip" in name or "femur" in name or "costal" in name:
                return "skeletal"
            elif "gluteus" in name or "autochthon" in name or "iliopsoas" in name:
                return "muscular"
            elif "kidney" in name or "bladder" in name or "adrenal" in name:
                return "urinary"
            elif "liver" in name or "stomach" in name or "pancreas" in name or \
                 "gallbladder" in name or "esophagus" in name or "bowel" in name or \
                 "duodenum" in name or "colon" in name:
                return "digestive"
            elif "heart" in name:
                return "cardiovascular"
            elif "brain" in name or "spinal_cord" in name:
                return "nervous"
            elif "lung" in name or "trachea" in name:
                return "respiratory"
            else:
                return "other"

        # Create dataframe
        df = pd.DataFrame({
            "x": coords_2d[:, 0],
            "y": coords_2d[:, 1],
            "class_name": class_names,
            "system": [get_system(name) for name in class_names],
            "depth": [class_depths[i] for i in range(70)],
        })

        # Create plot
        fig = px.scatter(
            df, x="x", y="y",
            color="system",
            symbol="system",
            hover_data=["class_name", "depth"],
            title="Label Embeddings t-SNE (colored by organ system)"
        )

        output_path = os.path.join(output_dir, "label_emb_tsne.html")
        fig.write_html(output_path)
        assert os.path.exists(output_path)
