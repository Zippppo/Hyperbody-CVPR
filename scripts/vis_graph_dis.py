"""Visualize the 70x70 graph distance matrix as an interactive heatmap."""
import json
import torch
import plotly.graph_objects as go

# Load data
dist = torch.load("Dataset/graph_distance_matrix.pt", map_location="cpu")
with open("Dataset/dataset_info.json") as f:
    info = json.load(f)

class_names = info["class_names"]  # length 70 (index 0 = inside_body_empty)
n = dist.shape[0]
labels = [f"{i}: {class_names[i]}" for i in range(n)]

fig = go.Figure(data=go.Heatmap(
    z=dist.numpy(),
    x=labels,
    y=labels,
    colorscale="Viridis",
    colorbar=dict(title="Graph Distance"),
    hovertemplate="Row: %{y}<br>Col: %{x}<br>Distance: %{z}<extra></extra>",
))

fig.update_layout(
    title="Graph Distance Matrix (70 × 70)",
    xaxis=dict(title="Class", tickangle=45, tickfont=dict(size=7)),
    yaxis=dict(title="Class", tickfont=dict(size=7), autorange="reversed"),
    width=1100,
    height=1000,
)

fig.write_html("scripts/graph_distance_heatmap.html")
print("Saved to outputs/graph_distance_heatmap.html")
