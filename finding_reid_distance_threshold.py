import json
from pathlib import Path
from typing import Generator, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm

from utils.processing import PersonRecognizer

BATCH_SIZE = 500


def batch_generator[T](
    iterable: Iterable[T], batch_size: int
) -> Generator[list[T], None, None]:
    """Yield batches from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_camera_id(filename: str) -> int:
    try:
        camera_info = filename.split("_")[1]
        return int(camera_info[1])
    except (IndexError, ValueError):
        return -1


def extract_person_id(filename: str) -> int:
    try:
        return int(filename.split("_")[0])
    except (IndexError, ValueError):
        return -1


class EmbeddingProcessor:
    """Handle embedding creation and storage"""

    def __init__(self, model_path: Path) -> None:
        self.model = PersonRecognizer(model_path)

    def create_embeddings(self, image_dir: Path, save_dir: Path) -> None:
        """Create and save embeddings from images"""
        save_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
        ]

        for image_path in tqdm(image_paths, desc="Creating embeddings"):
            embedding = self._infer(image_path)
            np.save(save_dir / f"{image_path.stem}.npy", embedding)

    def _infer(self, image_path: Path) -> np.ndarray:
        img = cv2.imread(str(image_path))
        return self.model.infer(img)


class DataLoader:
    """Handles loading of embeddings and metadata"""

    @staticmethod
    def load_embeddings(save_dir: Path) -> tuple[np.ndarray, list[str]]:
        """Load embeddings with metadata"""
        save_dir.mkdir(parents=True, exist_ok=True)
        all_files = [file for file in save_dir.iterdir() if file.suffix == ".npy"]
        all_embeddings: list[np.ndarray] = []
        all_metadata: list[str] = []

        for batch in tqdm(
            batch_generator(all_files, BATCH_SIZE),
            desc="Loading embeddings",
            total=(len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE,
        ):
            embeddings, metadata = DataLoader._load_batch(batch)
            all_embeddings.append(embeddings)
            all_metadata.extend(metadata)

        return np.vstack(all_embeddings), all_metadata

    @staticmethod
    def _load_batch(files: list[Path]) -> tuple[np.ndarray, list[str]]:
        """Load a single batch of embeddings"""
        results = []
        metadata = []
        for file in files:
            results.append(np.load(file))
            metadata.append(file.stem)
        return np.vstack(results), metadata


class ThresholdAnalyzer:
    """Handles threshold calculation and analysis"""

    @staticmethod
    def calculate_distances(
        query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, metric: str
    ) -> np.ndarray:
        """Calculate pairwise distances between embeddings"""
        distances = [
            batch
            for batch in tqdm(
                ThresholdAnalyzer._calculate_batch(
                    query_embeddings, gallery_embeddings, metric
                ),
                desc=f"Calculating {metric} distances",
                total=(len(query_embeddings) + BATCH_SIZE - 1) // BATCH_SIZE,
            )
        ]
        return np.vstack(distances)

    @staticmethod
    def _calculate_batch(
        query_embeddings: np.ndarray,
        gallery_embeddings: np.ndarray,
        metric: str,
    ) -> Generator[np.ndarray, None, None]:
        """Calculate distances in batches."""
        for i in range(0, len(query_embeddings), BATCH_SIZE):
            batch_query = query_embeddings[i : i + BATCH_SIZE]
            yield cdist(batch_query, gallery_embeddings, metric=metric)

    @staticmethod
    def create_ground_truth(
        query_metadata: list[str], gallery_metadata: list[str]
    ) -> np.ndarray:
        """Create ground truth matrix from metadata"""
        query_ids = np.array([extract_person_id(name) for name in query_metadata])
        gallery_ids = np.array([extract_person_id(name) for name in gallery_metadata])
        return query_ids[:, None] == gallery_ids[None, :]

    @staticmethod
    def find_optimal_threshold(
        distances: np.ndarray, ground_truth: np.ndarray
    ) -> tuple[float, np.ndarray, int, np.ndarray, np.ndarray, float]:
        """Find threshold using precision-recall curve"""
        scores = 1 / (1 + distances.ravel())
        precision, recall, thresholds = precision_recall_curve(
            ground_truth.ravel(), scores
        )
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_distance = (1 / thresholds[optimal_idx]) - 1
        return (
            optimal_distance,
            f1_scores,
            optimal_idx,
            precision,
            recall,
            auc(recall, precision),
        )

    @staticmethod
    def find_intersection_threshold(
        distances: np.ndarray, ground_truth: np.ndarray, n_bins: int = 100
    ) -> tuple[float, float, float]:
        """Find threshold at histogram intersection"""
        match_distances = distances[ground_truth]
        non_match_distances = distances[~ground_truth]

        min_dist = min(match_distances.min(), non_match_distances.min())
        max_dist = max(match_distances.max(), non_match_distances.max())

        match_hist, bin_edges = np.histogram(
            match_distances, bins=n_bins, range=(min_dist, max_dist), density=True
        )
        non_match_hist, _ = np.histogram(
            non_match_distances, bins=bin_edges, density=True
        )

        diff = match_hist - non_match_hist
        cross_points = np.where(diff[:-1] * diff[1:] <= 0)[0]
        mean_midpoint = (np.mean(match_distances) + np.mean(non_match_distances)) / 2

        if not cross_points.size:
            threshold = mean_midpoint
        else:
            best_cross_idx = cross_points[
                np.argmin(np.abs(bin_edges[cross_points] - mean_midpoint))
            ]
            x1, x2 = bin_edges[best_cross_idx], bin_edges[best_cross_idx + 1]
            y1, y2 = diff[best_cross_idx], diff[best_cross_idx + 1]

            threshold = (
                x1 + (x2 - x1) * (-y1) / (y2 - y1) if y1 != y2 else (x1 + x2) / 2
            )

        return (
            threshold,
            (distances[~ground_truth] < threshold).mean(),
            (distances[ground_truth] > threshold).mean(),
        )


class ResultVisualizer:
    """Handles result visualization"""

    @staticmethod
    def plot_comparison(
        distances: np.ndarray,
        ground_truth: np.ndarray,
        threshold: float,
        metric: str,
        save_path: Path,
        optimal_plot_type: bool = True,
        f1_scores: np.ndarray | None = None,
        optimal_idx: int | None = None,
        precision: np.ndarray | None = None,
        recall: np.ndarray | None = None,
        pr_auc: float | None = None,
        false_accept_rate: float | None = None,
        false_reject_rate: float | None = None,
    ) -> None:
        """Generic plotting function for different visualization types"""
        match_dist = distances[ground_truth]
        non_match_dist = distances[~ground_truth]

        if optimal_plot_type:
            plt.figure(figsize=(12, 6))
            ResultVisualizer._plot_histograms(
                plt.gca(), match_dist, non_match_dist, threshold, metric
            )
            if f1_scores is not None and optimal_idx is not None and pr_auc is not None:
                ResultVisualizer._add_optimal_metrics(f1_scores[optimal_idx], pr_auc)
            plt.savefig(save_path / f"{metric}_optimal_analysis.png")
            plt.close()

            if all(
                arg is not None
                for arg in [f1_scores, optimal_idx, precision, recall, pr_auc]
            ):
                plt.figure(figsize=(8, 6))
                ResultVisualizer._plot_pr_curve(
                    plt.gca(), f1_scores, optimal_idx, precision, recall, pr_auc
                )
                plt.savefig(save_path / f"{metric}_pr_curve.png")
                plt.close()
        else:
            plt.figure(figsize=(12, 6))
            ResultVisualizer._plot_histograms(
                plt.gca(), match_dist, non_match_dist, threshold, metric
            )
            if false_accept_rate is not None and false_reject_rate is not None:
                ResultVisualizer._add_error_rates(false_accept_rate, false_reject_rate)
            plt.savefig(save_path / f"{metric}_intersection_analysis.png")
            plt.close()

    @staticmethod
    def _plot_histograms(
        ax, match_data, non_match_data, threshold, metric, plot_bins: int = 50
    ):
        """Plot histogram subcomponent"""
        ax.hist(
            match_data,
            bins=plot_bins,
            alpha=0.5,
            label="Matches",
            density=True,
        )
        ax.hist(
            non_match_data,
            bins=plot_bins,
            alpha=0.5,
            label="Non-matches",
            density=True,
        )
        ax.axvline(
            threshold, color="r", linestyle="--", label=f"Threshold: {threshold:.3f}"
        )
        ax.set_title(f"Distance Distributions ({metric})")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _plot_pr_curve(ax, f1_scores, optimal_idx, precision, recall, pr_auc):
        """Plot precision-recall curve subcomponent"""
        optimal_recall = recall[optimal_idx]
        optimal_precision = precision[optimal_idx]

        ax.plot(recall, precision, label=f"PR Curve (AUC={pr_auc:.3f})", color="blue")
        ax.scatter(
            optimal_recall,
            optimal_precision,
            color="red",
            marker="o",
            label=f"Optimal F1: {f1_scores[optimal_idx]:.3f} (P={optimal_precision:.2f}, R={optimal_recall:.2f})",
        )

        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _add_optimal_metrics(f1_score: float, pr_auc: float):
        """Add F1 and AUC annotations"""
        plt.text(
            0.02,
            0.98,
            f"F1 Score: {f1_score:.3f}\nPR AUC: {pr_auc:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    @staticmethod
    def _add_error_rates(far: float, frr: float):
        """Add error rate annotations"""
        plt.text(
            0.02,
            0.98,
            f"FAR: {far:.3%}\nFRR: {frr:.3%}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )


def main():
    model_name = "openvino_smallest"

    # Initilize components
    model_path = Path(
        "./models/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml"
    )
    processor = EmbeddingProcessor(model_path)

    # Setup paths
    base_dir = Path("../mdx/Market-1501-v15.09.15").resolve()
    query_dir = base_dir / "query"
    test_dir = base_dir / "bounding_box_test"

    embeddings_dir = base_dir / "outputs" / model_name
    query_emb_dir = embeddings_dir / "embeddings_query"
    test_emb_dir = embeddings_dir / "embeddings_test"
    results_dir = embeddings_dir / "results"

    # Create embeddings if needed
    for src, dest in [(query_dir, query_emb_dir), (test_dir, test_emb_dir)]:
        if not dest.exists():
            processor.create_embeddings(src, dest)

    # Load data
    query_emb, query_meta = DataLoader.load_embeddings(query_emb_dir)
    test_emb, test_meta = DataLoader.load_embeddings(test_emb_dir)
    ground_truth = ThresholdAnalyzer.create_ground_truth(query_meta, test_meta)

    # Extract camera IDs
    query_cameras = np.array([extract_camera_id(name) for name in query_meta])
    test_cameras = np.array([extract_camera_id(name) for name in test_meta])

    # Process different distance metrics
    metrics = ["euclidean", "cosine"]
    results = {}

    for metric in metrics:
        print(f"\nProcessing {metric} distance:")

        # Calculate distances
        distances = ThresholdAnalyzer.calculate_distances(query_emb, test_emb, metric)

        # Define scenarios
        scenarios = [
            ("same_camera", query_cameras[:, None] == test_cameras[None, :]),
            ("cross_camera", query_cameras[:, None] != test_cameras[None, :]),
            ("all", np.ones_like(ground_truth, dtype=bool)),
        ]

        for scenario_name, mask in scenarios:
            print(f"  Scenarios: {scenario_name}")
            scenario_distances = distances[mask]
            scenario_gt = ground_truth[mask]

            if not scenario_distances.size:
                print(f"    No data for {scenario_name}, skipping.")
                continue

            # Calculate thresholds
            opt_threshold, f1_scores, opt_idx, precision, recall, pr_auc = (
                ThresholdAnalyzer.find_optimal_threshold(
                    scenario_distances, scenario_gt
                )
            )

            print(f"Optimal {metric} threshold: {opt_threshold:.3f}")
            print(f"F1 Score: {f1_scores[opt_idx]:.3f}")
            print(f"PR AUC: {pr_auc:.3f}")

            intersect_threshold, far, frr = (
                ThresholdAnalyzer.find_intersection_threshold(
                    scenario_distances, scenario_gt
                )
            )

            print(f"Intersection {metric} threshold: {intersect_threshold:.3f}")
            print(f"False Accept Rate: {far:.3%}")
            print(f"False Reject Rate: {frr:.3%}")

            # Generate visualizations
            scenario_save_path = results_dir / scenario_name
            scenario_save_path.mkdir(parents=True, exist_ok=True)

            ResultVisualizer.plot_comparison(
                scenario_distances,
                scenario_gt,
                opt_threshold,
                metric,
                scenario_save_path,
                optimal_plot_type=True,
                f1_scores=f1_scores,
                optimal_idx=opt_idx,
                precision=precision,
                recall=recall,
                pr_auc=pr_auc,
            )
            ResultVisualizer.plot_comparison(
                scenario_distances,
                scenario_gt,
                intersect_threshold,
                metric,
                scenario_save_path,
                optimal_plot_type=False,
                false_accept_rate=far,
                false_reject_rate=frr,
            )

            # Store results
            if metric not in results:
                results[metric] = {}
            results[metric][scenario_name] = {
                "optimal_threshold": opt_threshold,
                "f1_score": f1_scores[opt_idx],
                "pr_auc": pr_auc,
                "intersection_threshold": intersect_threshold,
                "false_accept_rate": far,
                "false_reject_rate": frr,
            }

    # Save final results
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
