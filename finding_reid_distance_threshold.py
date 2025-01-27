from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from utils.processing import PersonRecognizer

MODEL_NAME = "openvino"
MODEL = PersonRecognizer(
    "./models/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml"
)


# Assume the infer function is provided to extract embeddings
def infer(image_path: Path):
    img = cv2.imread(str(image_path))
    return MODEL.infer(img)


# Function to create embeddings and save them file by file
def create_embeddings_filewise(image_dir: Path, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    def process_image(image_path: Path):
        if image_path.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            embedding = infer(image_path)
            embedding_path = save_dir / f"{image_path.stem}.npy"
            np.save(embedding_path, embedding)

    with Pool(cpu_count()) as pool:
        pool.map(process_image, image_dir.iterdir())

    print(f"Saved individual embeddings in {save_dir}")


# Function to load a single embedding file
def load_file(file: Path):
    if file.suffix == ".npy":
        return np.load(file), file.stem
    return None


# Function to load embeddings and metadata from saved files
def load_embeddings(save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    with Pool(cpu_count()) as pool:
        results = pool.map(load_file, save_dir.iterdir())

    embeddings, metadata = zip(*filter(None, results))
    embeddings = np.vstack(embeddings)
    return embeddings, metadata


# Function to calculate pairwise distances
def calculate_distances(query_embeddings, gallery_embeddings, metric="euclidean"):
    distances = cdist(query_embeddings, gallery_embeddings, metric=metric)
    return distances


# Function to extract camera ID from metadata
def extract_camera_id(filename):
    try:
        camera_info = filename.split("_")[1]  # Extract "cXsY" part
        camera_id = int(camera_info[1])  # Extract numeric part after 'c'
        return camera_id
    except (IndexError, ValueError):
        return -1  # Return -1 if parsing fails


# Function to filter matches based on camera ID
def filter_by_camera(metadata_query, metadata_gallery):
    query_camera_ids = [extract_camera_id(name) for name in metadata_query]
    gallery_camera_ids = [extract_camera_id(name) for name in metadata_gallery]

    matches_same_camera = []
    matches_across_cameras = []

    for i, q_cam in enumerate(query_camera_ids):
        for j, g_cam in enumerate(gallery_camera_ids):
            if q_cam == g_cam:
                matches_same_camera.append((i, j))
            else:
                matches_across_cameras.append((i, j))

    return matches_same_camera, matches_across_cameras


# Function to plot results
def plot_results(distances, matches, title, ax):
    filtered_distances = [distances[q, g] for q, g in matches]
    ax.hist(filtered_distances, bins=50, alpha=0.75, color="blue")
    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")


# Main script
if __name__ == "__main__":
    base_dir = Path("../mdx/Market-1501-v15.09.15").resolve()
    query_dir = base_dir / "query"
    test_dir = base_dir / "bounding_box_test"

    # Create directories for embeddings
    embeddings_base_dir = base_dir / MODEL_NAME
    query_embeddings_dir = embeddings_base_dir / "embeddings_query"
    test_embeddings_dir = embeddings_base_dir / "embeddings_test"

    embeddings_base_dir.mkdir(parents=True, exist_ok=True)

    if not query_embeddings_dir.exists():
        create_embeddings_filewise(query_dir, query_embeddings_dir)
    if not test_embeddings_dir.exists():
        create_embeddings_filewise(test_dir, test_embeddings_dir)

    # Load embeddings and metadata
    query_embeddings, query_metadata = load_embeddings(query_embeddings_dir)
    test_embeddings, test_metadata = load_embeddings(test_embeddings_dir)
    print("Loaded all embeddings.")

    # Calculate distances
    metrics = ["euclidean", "cosine"]
    for metric in metrics:
        distances = calculate_distances(
            query_embeddings, test_embeddings, metric=metric
        )
        print(f"Calculated {metric} distances.")

        # Prepare plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)

        # 1 & 2. Match same camera and across cameras
        matches_same_camera, matches_across_cameras = filter_by_camera(
            query_metadata, test_metadata
        )
        plot_results(distances, matches_same_camera, "Matches: Same Camera", axes[0])
        plot_results(
            distances, matches_across_cameras, "Matches: Across Cameras", axes[1]
        )
        print("Filtered by cameras.")

        # 3. Match all query to test
        matches_all = [
            (i, j)
            for i in range(len(query_metadata))
            for j in range(len(test_metadata))
        ]
        plot_results(
            distances,
            matches_all,
            "Matches: All Query to Test",
            axes[2],
        )
        print("All cameras.")

        # Save and show the combined plot
        combined_plot_path = embeddings_base_dir / f"{metric}_combined_plot.png"
        plt.savefig(combined_plot_path)
        plt.show()
