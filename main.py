import numpy as np
from PIL import Image

def load_image(image_path):
    """Load an image and convert it to a numpy array."""
    image = Image.open(image_path)
    return np.asarray(image) / 255.0, image.size  # Normalize to [0, 1] and return original size


def save_image(image_array, original_size, output_path):
    """Save a numpy array as an image without black padding and resize to original size."""
    image = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image).resize(original_size, Image.LANCZOS)
    image.save(output_path)


def resize_image(image_array, target_size):
    """Resize an image array to a target size."""
    return np.asarray(Image.fromarray((image_array * 255).astype(np.uint8)).resize(target_size, Image.LANCZOS)) / 255.0


def sliced_wasserstein_texture_mixing(target, source, num_projections=100, num_iterations=5):
    """
    Perform texture mixing between the source and target images using
    the Sliced Wasserstein Distance (SWD).

    Args:
        source: Source image as a numpy array of shape (H, W, 3).
        target: Target image as a numpy array of shape (H, W, 3).
        original_size: Original size of the target image.
        num_projections: Number of random projections for SWD.
        num_iterations: Number of iterations to refine the mixing.

    Returns:
        Mixed image as a numpy array.
    """
    
    h, w, c = target.shape

    # Reshape to 2D arrays
    source_flat = source.reshape(-1, 3)
    target_flat = target.reshape(-1, 3)

    for _ in range(num_iterations):
        # Generate random projections
        projections = np.random.randn(num_projections, 3)
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)

        for proj in projections:
            # Project source and target onto random direction
            source_proj = source_flat @ proj
            target_proj = target_flat @ proj

            # Sort projections
            target_sorted = np.sort(target_proj)

            # Compute the Sliced Wasserstein Distance and update source towards target
            source_idx = np.argsort(source_proj)
            transported_proj = np.zeros_like(source_proj)
            transported_proj[source_idx] = target_sorted

            # Update the source_flat towards target using gradient flow
            source_flat += np.outer((transported_proj - source_proj), proj)

    mixed = source_flat.reshape(h, w, c)

    # Clip values to [0, 1]
    return np.clip(mixed, 0, 1)


# Load source and target images
source_image_path = "f.jpg"
target_image_path = "g.jpg"

source_image, source_size = load_image(source_image_path)
target_image, target_size = load_image(target_image_path)

# Resize images to the same size for processing
common_size = (1000, 1000)  # Define a common size for computation
source_resized = resize_image(source_image, common_size)
target_resized = resize_image(target_image, common_size)

# Apply texture mixing
mixed_image = sliced_wasserstein_texture_mixing(source_resized, target_resized)

# Save the result
output_path = "output_image.png"

save_image(mixed_image, target_size, output_path)

print("Texture mixing complete. Result saved at:", output_path)