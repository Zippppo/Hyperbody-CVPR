"""Integration tests for scripts/precompute_graph_distance.py."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _write_npz(path: Path, voxel_labels: np.ndarray) -> None:
    """Create a minimal .npz file with required fields."""
    sensor_pc = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        sensor_pc=sensor_pc,
        grid_world_min=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        grid_voxel_size=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        voxel_labels=voxel_labels.astype(np.int64),
    )


class TestPrecomputeGraphDistanceScript:
    """Integration tests for scripts/precompute_graph_distance.py."""

    @pytest.fixture
    def synthetic_env(self, tmp_path):
        """Create synthetic dataset, split, info, tree and output paths."""
        data_dir = tmp_path / "voxel_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        lbl_1 = np.zeros((12, 12, 12), dtype=np.int64)
        lbl_1[2:6, 2:8, 2:8] = 1
        lbl_1[6:10, 2:8, 2:8] = 2

        lbl_2 = np.zeros((12, 12, 12), dtype=np.int64)
        lbl_2[2:10, 2:10, 2:10] = 1
        lbl_2[4:8, 4:8, 4:8] = 2

        _write_npz(data_dir / "sample_1.npz", lbl_1)
        _write_npz(data_dir / "sample_2.npz", lbl_2)

        split_path = tmp_path / "dataset_split.json"
        split_path.write_text(
            json.dumps({"train": ["sample_1.npz", "sample_2.npz"], "val": [], "test": []}),
            encoding="utf-8",
        )

        class_names = ["inside_body_empty", "organ_a", "organ_b"]
        info_path = tmp_path / "dataset_info.json"
        info_path.write_text(json.dumps({"class_names": class_names}), encoding="utf-8")

        tree = {
            "human_body": {
                "synthetic_system": {
                    "organ_a": "organ_a",
                    "organ_b": "organ_b",
                }
            }
        }
        tree_path = tmp_path / "tree.json"
        tree_path.write_text(json.dumps(tree), encoding="utf-8")

        output_dir = tmp_path / "outputs"

        return {
            "data_dir": data_dir,
            "split_path": split_path,
            "info_path": info_path,
            "tree_path": tree_path,
            "output_dir": output_dir,
            "class_names": class_names,
            "repo_root": Path(__file__).resolve().parents[2],
        }

    def _run_script(self, synthetic_env, output_dir=None, extra_args=None):
        """Run the script and return CompletedProcess."""
        output_dir = output_dir or synthetic_env["output_dir"]
        cmd = [
            sys.executable,
            "scripts/precompute_graph_distance.py",
            "--output-dir",
            str(output_dir),
            "--tree-file",
            str(synthetic_env["tree_path"]),
            "--data-dir",
            str(synthetic_env["data_dir"]),
            "--split-file",
            str(synthetic_env["split_path"]),
            "--dataset-info",
            str(synthetic_env["info_path"]),
            "--volume-size",
            "12",
            "12",
            "12",
            "--dilation-radius",
            "1",
            "--lambda",
            "1.0",
            "--epsilon",
            "0.01",
            "--class-batch-size",
            "2",
            "--num-workers",
            "0",
        ]
        if extra_args:
            cmd.extend(extra_args)

        return subprocess.run(
            cmd,
            cwd=str(synthetic_env["repo_root"]),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_end_to_end_produces_both_files(self, synthetic_env):
        """Script should produce both contact_matrix.pt and graph_distance_matrix.pt."""
        result = self._run_script(synthetic_env)

        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        assert (synthetic_env["output_dir"] / "contact_matrix.pt").exists()
        assert (synthetic_env["output_dir"] / "graph_distance_matrix.pt").exists()

    def test_output_dir_auto_created(self, synthetic_env):
        """Script should create output directory if it does not exist."""
        output_dir = synthetic_env["output_dir"] / "nested" / "graph"
        result = self._run_script(synthetic_env, output_dir=output_dir)

        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        assert output_dir.exists()
        assert (output_dir / "graph_distance_matrix.pt").exists()

    def test_graph_distance_shape_and_dtype(self, synthetic_env):
        """Output tensors should have shape (num_classes, num_classes) and dtype float32."""
        result = self._run_script(synthetic_env)
        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

        graph = torch.load(synthetic_env["output_dir"] / "graph_distance_matrix.pt")
        contact = torch.load(synthetic_env["output_dir"] / "contact_matrix.pt")

        assert graph.shape == (3, 3)
        assert contact.shape == (3, 3)
        assert graph.dtype == torch.float32
        assert contact.dtype == torch.float32

    def test_graph_distance_diagonal_is_zero(self, synthetic_env):
        """Graph distance matrix diagonal should be all zeros."""
        result = self._run_script(synthetic_env)
        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

        graph = torch.load(synthetic_env["output_dir"] / "graph_distance_matrix.pt")
        assert torch.allclose(torch.diag(graph), torch.zeros(3))

    def test_graph_distance_leq_tree_distance(self, synthetic_env):
        """graph_distance[i,j] <= tree_distance[i,j] for all pairs."""
        from data.organ_hierarchy import compute_tree_distance_matrix

        result = self._run_script(synthetic_env)
        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

        graph = torch.load(synthetic_env["output_dir"] / "graph_distance_matrix.pt")
        tree = compute_tree_distance_matrix(
            str(synthetic_env["tree_path"]),
            synthetic_env["class_names"],
        )
        assert torch.all(graph <= tree + 1e-6)

    def test_lambda_epsilon_affect_result(self, synthetic_env):
        """Different lambda/epsilon should produce different graph distances."""
        output_a = synthetic_env["output_dir"] / "run_a"
        output_b = synthetic_env["output_dir"] / "run_b"

        result_a = self._run_script(
            synthetic_env,
            output_dir=output_a,
            extra_args=["--lambda", "0.1", "--epsilon", "0.01"],
        )
        assert result_a.returncode == 0, f"stderr:\n{result_a.stderr}\nstdout:\n{result_a.stdout}"

        result_b = self._run_script(
            synthetic_env,
            output_dir=output_b,
            extra_args=["--lambda", "10.0", "--epsilon", "1.0"],
        )
        assert result_b.returncode == 0, f"stderr:\n{result_b.stderr}\nstdout:\n{result_b.stdout}"

        graph_a = torch.load(output_a / "graph_distance_matrix.pt")
        graph_b = torch.load(output_b / "graph_distance_matrix.pt")

        assert not torch.allclose(graph_a, graph_b)

    def test_contact_matrix_flag_skips_computation(self, synthetic_env):
        """--contact-matrix flag should load existing file and skip dataset traversal."""
        first_output = synthetic_env["output_dir"] / "first"
        second_output = synthetic_env["output_dir"] / "second"

        first = self._run_script(synthetic_env, output_dir=first_output)
        assert first.returncode == 0, f"stderr:\n{first.stderr}\nstdout:\n{first.stdout}"

        contact_path = first_output / "contact_matrix.pt"
        second = self._run_script(
            synthetic_env,
            output_dir=second_output,
            extra_args=["--contact-matrix", str(contact_path)],
        )
        assert second.returncode == 0, f"stderr:\n{second.stderr}\nstdout:\n{second.stdout}"
        assert "Loaded contact matrix from" in second.stdout

        graph_first = torch.load(first_output / "graph_distance_matrix.pt")
        graph_second = torch.load(second_output / "graph_distance_matrix.pt")
        assert torch.allclose(graph_first, graph_second)

    def test_stdout_contains_statistics(self, synthetic_env):
        """Script should print statistics: classes count, non-zero contacts, shortened pairs."""
        result = self._run_script(synthetic_env)

        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        assert "Classes: 3" in result.stdout
        assert "Non-zero contacts" in result.stdout
        assert "Shortened pairs" in result.stdout
        combined_output = result.stdout + result.stderr
        assert "Contact matrix:" in combined_output
        assert "100%" in combined_output



def test_config_has_graph_distance_matrix_field():
    """Config should have graph_distance_matrix field with empty string default."""
    from config import Config

    cfg = Config()
    assert hasattr(cfg, "graph_distance_matrix")
    assert cfg.graph_distance_matrix == ""


class TestTrainGraphModeLoading:
    """Test that train.py graph mode requires precomputed file."""

    def test_graph_mode_missing_file_raises_error(self, tmp_path):
        """When graph_distance_matrix is empty or missing, should raise FileNotFoundError."""
        import logging

        from train import load_precomputed_graph_distance_matrix

        logger = logging.getLogger("test_train_graph_mode")

        with pytest.raises(FileNotFoundError):
            load_precomputed_graph_distance_matrix("", logger)

        with pytest.raises(FileNotFoundError):
            load_precomputed_graph_distance_matrix(str(tmp_path / "missing.pt"), logger)

    def test_graph_mode_loads_precomputed_file(self, tmp_path):
        """When graph_distance_matrix points to valid file, should load successfully."""
        import logging

        from train import load_precomputed_graph_distance_matrix

        expected = torch.tensor(
            [
                [0.0, 2.0, 4.0],
                [2.0, 0.0, 3.0],
                [4.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        matrix_path = tmp_path / "graph_distance_matrix.pt"
        torch.save(expected, matrix_path)

        logger = logging.getLogger("test_train_graph_mode")
        loaded = load_precomputed_graph_distance_matrix(str(matrix_path), logger)

        assert torch.allclose(loaded, expected)
        assert loaded.dtype == torch.float32

    def test_train_py_no_longer_imports_removed_functions(self):
        """train.py should not import compute_contact_matrix_from_dataset etc."""
        import ast

        train_path = Path(__file__).resolve().parents[2] / "train.py"
        source = train_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "data.spatial_adjacency":
                for alias in node.names:
                    imported_names.add(alias.name)

        assert "compute_contact_matrix_from_dataset" not in imported_names
        assert "compute_graph_distance_matrix" not in imported_names
        assert "infer_ignored_spatial_class_indices" not in imported_names
