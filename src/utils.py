import numpy as np
from PIL import Image


def preprocess_digit_image(image_path, invert=False):
    """
    Preprocess an external handwritten digit image for MNIST CNN prediction.

    Steps:
    1. Load image
    2. Convert to grayscale
    3. Resize to 28x28
    4. Normalize pixel values to 0-1
    5. Optionally invert colors
    6. Reshape to CNN input format

    Parameters:
    image_path: str
        Path to the image file.
    invert: bool
        If True, invert image colors.
        Use this when the image has black digit on white background.

    Returns:
    processed_image: numpy array
        Image reshaped to (1, 28, 28, 1)
    display_image: numpy array
        Image before reshape, useful for visualization.
    """

    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image).astype("float32") / 255.0

    if invert:
        image_array = 1.0 - image_array

    processed_image = image_array.reshape(1, 28, 28, 1)

    return processed_image, image_array