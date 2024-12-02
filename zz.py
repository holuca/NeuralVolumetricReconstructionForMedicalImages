import numpy as np
import matplotlib.pyplot as plt

projections = np.load("./logs/chest_pretrained_parallel_chest/density.npy")
print("Shape of projections:", projections.shape)

def visualize_projections(projections, indices):
    """
    Visualize specific projections from the 3D array.

    :param projections: 3D numpy array of shape (num_projections, height, width)
    :param indices: List of indices of the projections to visualize
    """
    num_images = len(indices)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # Ensure axes is iterable even for a single image
    if num_images == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.imshow(projections[idx], cmap="gray")
        ax.set_title(f"Projection {idx}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

indices_to_view = [0, 10, 20, 30, 100]  # Replace with the indices you want to view
visualize_projections(projections, indices_to_view)