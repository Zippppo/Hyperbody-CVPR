"""
Track and visualize label embedding evolution during training.

Generates:
- JSON history file with tangent vectors and Poincaré positions per epoch
- PNG static images for each epoch
- HTML animation with slider and play button
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.hyperbolic.label_embedding import LorentzLabelEmbedding

import numpy as np
import torch
from sklearn.decomposition import PCA

from models.hyperbolic.lorentz_ops import lorentz_to_poincare, distance_to_origin


SYSTEM_COLORS = {
    "skeletal": "#E74C3C",      # red
    "muscular": "#3498DB",      # blue
    "digestive": "#2ECC71",     # green
    "respiratory": "#9B59B6",   # purple
    "cardiovascular": "#E67E22", # orange
    "urinary": "#1ABC9C",       # cyan
    "nervous": "#F1C40F",       # yellow
    "other": "#95A5A6"          # gray
}


class EmbeddingTracker:
    """
    Track label embedding evolution during training.

    Records tangent vectors and Poincaré positions at each epoch,
    generates PNG visualizations and HTML animation.
    """

    def __init__(
        self,
        model_name: str,
        class_names: List[str],
        class_to_system: Dict[int, str],
        output_dir: str = "vis_res",
        curv: float = 1.0
    ):
        """
        Args:
            model_name: Name for output subdirectory
            class_names: List of class names (index = class_idx)
            class_to_system: Mapping from class index to organ system name
            output_dir: Base output directory
            curv: Hyperbolic curvature parameter
        """
        self.model_name = model_name
        self.class_names = class_names
        self.class_to_system = class_to_system
        self.output_dir = os.path.join(output_dir, model_name)
        self.curv = curv

        self.num_classes = len(class_names)
        self.pca_transform: Optional[np.ndarray] = None  # 2 x embed_dim

        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(
        self,
        epoch: int,
        label_embedding: "LorentzLabelEmbedding"
    ) -> None:
        """
        Record embeddings and generate visualizations.

        Call after each epoch (including epoch=0 before training).

        Args:
            epoch: Current epoch number
            label_embedding: LorentzLabelEmbedding module
        """
        # Extract tangent vectors
        with torch.no_grad():
            tangent_vectors = label_embedding.tangent_embeddings.detach().cpu().numpy()
            lorentz_positions = label_embedding().detach()
            poincare_positions = lorentz_to_poincare(lorentz_positions, self.curv).cpu().numpy()
            distances = distance_to_origin(lorentz_positions, self.curv).cpu().numpy()

        # Check for NaN/Inf
        has_nan = (
            np.any(np.isnan(tangent_vectors)) or
            np.any(np.isinf(tangent_vectors)) or
            np.any(np.isnan(poincare_positions)) or
            np.any(np.isinf(poincare_positions))
        )

        # PCA for 2D projection
        if epoch == 0:
            # Fit PCA on epoch 0 and store transform
            pca = PCA(n_components=2)
            pca.fit(poincare_positions)
            self.pca_transform = pca.components_  # 2 x embed_dim
            self._save_metadata(tangent_vectors.shape[1])
        elif self.pca_transform is None:
            raise RuntimeError(
                f"on_epoch_end called with epoch={epoch} before epoch=0. "
                "Must call on_epoch_end(epoch=0, ...) first to initialize PCA transform."
            )

        poincare_2d = poincare_positions @ self.pca_transform.T  # N x 2

        # Record to JSON
        self._append_epoch_to_json(
            epoch, tangent_vectors, poincare_positions, distances, has_nan
        )

        # Generate PNG
        self._generate_png(epoch, poincare_2d, has_nan)

        # Update HTML animation
        self._generate_animation_html()

    def _save_metadata(self, embed_dim: int) -> None:
        """Save metadata to JSON file (called once at epoch 0)."""
        json_path = os.path.join(self.output_dir, "embedding_history.json")

        metadata = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "embed_dim": embed_dim,
            "curv": self.curv,
            "class_names": self.class_names,
            "class_to_system": {str(k): v for k, v in self.class_to_system.items()},
            "pca_components": self.pca_transform.tolist()
        }

        data = {"metadata": metadata, "epochs": []}
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def _append_epoch_to_json(
        self,
        epoch: int,
        tangent_vectors: np.ndarray,
        poincare_positions: np.ndarray,
        distances: np.ndarray,
        has_nan: bool
    ) -> None:
        """Append epoch data to JSON file."""
        json_path = os.path.join(self.output_dir, "embedding_history.json")

        with open(json_path, "r") as f:
            data = json.load(f)

        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "tangent_vectors": tangent_vectors.tolist(),
            "poincare_positions": poincare_positions.tolist(),
            "distances_to_origin": distances.tolist(),
            "has_nan": bool(has_nan)  # Convert numpy.bool_ to Python bool
        }

        data["epochs"].append(epoch_data)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_png(
        self,
        epoch: int,
        poincare_2d: np.ndarray,
        has_nan: bool
    ) -> None:
        """Generate PNG visualization for current epoch."""
        import plotly.graph_objects as go

        # Get colors for each point
        colors = [SYSTEM_COLORS.get(self.class_to_system.get(i, "other"), "#95A5A6")
                  for i in range(self.num_classes)]

        # Create figure
        fig = go.Figure()

        # Draw Poincaré ball boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Plot points
        fig.add_trace(go.Scatter(
            x=poincare_2d[:, 0],
            y=poincare_2d[:, 1],
            mode='markers+text',
            marker=dict(size=10, color=colors),
            text=[str(i) for i in range(self.num_classes)],
            textposition='top center',
            textfont=dict(size=8),
            hovertext=[f"{i}: {self.class_names[i]} ({self.class_to_system.get(i, 'other')})"
                       for i in range(self.num_classes)],
            hoverinfo='text',
            showlegend=False
        ))

        # Title
        title = f"Epoch {epoch}"
        if has_nan:
            title += " [WARNING: NaN detected]"

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis=dict(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-1.2, 1.2]),
            width=800,
            height=800,
            showlegend=False
        )

        # Try to save as PNG, fall back to HTML if kaleido is not available
        png_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}.png")
        try:
            fig.write_image(png_path)
        except (ValueError, ImportError, RuntimeError):
            # Fall back to saving as HTML if PNG export fails (kaleido/Chrome not installed)
            html_path = os.path.join(self.output_dir, f"epoch_{epoch:03d}.html")
            fig.write_html(html_path)

    def _generate_animation_html(self) -> None:
        """Generate HTML animation with all epochs so far."""
        import plotly.graph_objects as go

        json_path = os.path.join(self.output_dir, "embedding_history.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        epochs_data = data["epochs"]
        if not epochs_data:
            return

        # Get colors
        colors = [SYSTEM_COLORS.get(self.class_to_system.get(i, "other"), "#95A5A6")
                  for i in range(self.num_classes)]

        # Create frames
        frames = []
        for epoch_data in epochs_data:
            epoch = epoch_data["epoch"]
            poincare_full = np.array(epoch_data["poincare_positions"])
            poincare_2d = poincare_full @ self.pca_transform.T

            frame = go.Frame(
                data=[go.Scatter(
                    x=poincare_2d[:, 0],
                    y=poincare_2d[:, 1],
                    mode='markers',
                    marker=dict(size=10, color=colors),
                    hovertext=[f"{i}: {self.class_names[i]} ({self.class_to_system.get(i, 'other')})"
                               for i in range(self.num_classes)],
                    hoverinfo='text'
                )],
                name=str(epoch)
            )
            frames.append(frame)

        # Initial data (epoch 0)
        first_poincare = np.array(epochs_data[0]["poincare_positions"])
        first_2d = first_poincare @ self.pca_transform.T

        fig = go.Figure(
            data=[
                # Poincaré ball boundary
                go.Scatter(
                    x=np.cos(np.linspace(0, 2 * np.pi, 100)),
                    y=np.sin(np.linspace(0, 2 * np.pi, 100)),
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                # Points
                go.Scatter(
                    x=first_2d[:, 0],
                    y=first_2d[:, 1],
                    mode='markers',
                    marker=dict(size=10, color=colors),
                    hovertext=[f"{i}: {self.class_names[i]} ({self.class_to_system.get(i, 'other')})"
                               for i in range(self.num_classes)],
                    hoverinfo='text',
                    showlegend=False
                )
            ],
            frames=frames
        )

        # Add slider and play button
        fig.update_layout(
            title=dict(text="Label Embedding Evolution", x=0.5),
            xaxis=dict(range=[-1.2, 1.2], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-1.2, 1.2]),
            width=900,
            height=800,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.1,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {
                                "frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300}
                            }]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }]
                        )
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=16),
                        prefix="Epoch: ",
                        visible=True,
                        xanchor="right"
                    ),
                    transition=dict(duration=300),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[[str(e["epoch"])], {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }],
                            label=str(e["epoch"]),
                            method="animate"
                        )
                        for e in epochs_data
                    ]
                )
            ]
        )

        # Add legend for organ systems
        for system, color in SYSTEM_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=system,
                showlegend=True
            ))

        html_path = os.path.join(self.output_dir, "animation.html")
        fig.write_html(html_path)
