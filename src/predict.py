import argparse

from utils import DEFAULT_MODEL_PATH, load_digit_image, resolve_project_path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image.")
    parser.add_argument("--image", required=True, help="Path to a PNG/JPG image containing one digit.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH.relative_to(DEFAULT_MODEL_PATH.parents[1])),
        help="Path to the trained .keras model.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert image colors for black digits on white backgrounds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    import numpy as np
    from tensorflow.keras.models import load_model

    model_path = resolve_project_path(args.model_path)
    image_path = resolve_project_path(args.image)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Train the model first with: python3 src/train.py"
        )

    print(f"Loading model from: {model_path}")
    model = load_model(model_path)

    image_array = load_digit_image(image_path, invert=args.invert)
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
