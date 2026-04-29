# Visualization Scripts

Common graph/contact visualizations:

```bash
python scripts/visualization/visualize_contact_matrix.py --dataset-dir S2I_Dataset
python scripts/visualization/visualize_graph_distance.py --dataset-dir S2I_Dataset
python scripts/visualization/visualize_graph_minus_tree.py --dataset-dir S2I_Dataset
python scripts/visualization/visualize_contact_distribution.py --dataset-dir S2I_Dataset
```

Outputs are written to `outputs/visualization/` with the dataset directory name
as the filename prefix.

Notes:
- `visualize_contact_matrix_paper.py` uses old paper-specific class groups and
  is mainly intended for the original 70-class `Dataset`.
- `visualize_pipeline_figure.py` and `visualize_organ_dilation_3d.py` are
  qualitative 3D/pipeline figure utilities, not graph-distance diagnostics.
