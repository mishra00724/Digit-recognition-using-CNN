import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from utils import preprocess_digit_image


def predict_digit(image_path, model_path, invert=False):
    """
    Load a trained MNIST CNN model and predict the digit from an image.
    """

    print("Loading model from:", model_path)

    model = load_model(model_path)

    processed_image, display_image = preprocess_digit_image(
        image_path=image_path,
        invert=invert
    )

    prediction = model.predict(processed_image)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    print("Predicted Digit:", predicted_digit)
    print("Confidence:", round(confidence, 4))

    plt.imshow(display_image, cmap="gray")
    plt.title(f"Predicted Digit: {predicted_digit} | Confidence: {confidence:.4f}")
    plt.axis("off")
    plt.show()

    return predicted_digit, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict handwritten digit from an image using trained MNIST CNN model."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input digit image."
    )

    parser.add_argument(
        "--model",
        default="models/mnist_cnn_model.keras",
        help="Path to trained Keras model."
    )

    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert image colors. Use this for black digit on white background."
    )

    args = parser.parse_args()

    predict_digit(
        image_path=args.image,
        model_path=args.model,
        invert=args.invert
    )


if __name__ == "__main__":
    main()