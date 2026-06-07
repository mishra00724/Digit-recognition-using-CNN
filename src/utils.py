from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mnist_cnn_model.keras"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def resolve_project_path(path_value):
    """Resolve relative paths from the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_mnist_data(validation_size=0.1, random_state=42):
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    x_train, x_val, y_train_cat, y_val_cat = train_test_split(
        x_train,
        y_train_cat,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    return x_train, x_val, x_test, y_train_cat, y_val_cat, y_test_cat, y_test


def build_mnist_cnn():
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Conv2D(filters=32, kernel_size=(4, 4), activation="relu", input_shape=(28, 28, 1)),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def save_training_history(history, results_dir):
    import pandas as pd

    results_dir = ensure_directory(results_dir)
    history_path = results_dir / "training_history.csv"
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    return history_path


def save_training_curves(history, results_dir):
    import matplotlib.pyplot as plt

    results_dir = ensure_directory(results_dir)
    history_data = history.history

    loss_path = results_dir / "loss_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(history_data["loss"], label="Training Loss")
    plt.plot(history_data["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()

    accuracy_path = results_dir / "accuracy_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(history_data["accuracy"], label="Training Accuracy")
    plt.plot(history_data["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_path, dpi=150)
    plt.close()

    return loss_path, accuracy_path


def save_evaluation_outputs(y_true, y_pred, results_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

    results_dir = ensure_directory(results_dir)

    report_path = results_dir / "classification_report.txt"
    report = classification_report(y_true, y_pred, digits=4)
    report_path.write_text(report, encoding="utf-8")

    matrix_path = results_dir / "confusion_matrix.png"
    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 8))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=150)
    plt.close(fig)

    return report_path, matrix_path


def load_digit_image(image_path, invert=False):
    import numpy as np
    from PIL import Image, ImageOps

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("L")
    image = ImageOps.fit(image, (28, 28), method=Image.Resampling.LANCZOS)

    if invert:
        image = ImageOps.invert(image)

    image_array = np.asarray(image).astype("float32") / 255.0
    return image_array.reshape((1, 28, 28, 1))
