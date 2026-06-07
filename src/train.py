import argparse

from utils import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_DIR,
    build_mnist_cnn,
    ensure_directory,
    prepare_mnist_data,
    resolve_project_path,
    save_evaluation_outputs,
    save_training_curves,
    save_training_history,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CNN for MNIST handwritten digit classification."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH.relative_to(DEFAULT_MODEL_PATH.parents[1])),
        help="Where to save the trained .keras model.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR.relative_to(DEFAULT_RESULTS_DIR.parents[0])),
        help="Directory where evaluation outputs are saved.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Fraction of the training data reserved for validation.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience based on validation loss.",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for the split.")
    return parser.parse_args()


def main():
    args = parse_args()

    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping

    model_path = resolve_project_path(args.model_path)
    results_dir = resolve_project_path(args.results_dir)
    ensure_directory(model_path.parent)
    ensure_directory(results_dir)

    print("Loading and preprocessing MNIST data...")
    x_train, x_val, x_test, y_train, y_val, y_test_cat, y_test = prepare_mnist_data(
        validation_size=args.validation_size,
        random_state=args.random_seed,
    )

    print("Building CNN model...")
    model = build_mnist_cnn()
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    print("Evaluating on the separate MNIST test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    probabilities = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    print("Saving model and evaluation outputs...")
    model.save(model_path)
    history_path = save_training_history(history, results_dir)
    loss_path, accuracy_path = save_training_curves(history, results_dir)
    report_path, matrix_path = save_evaluation_outputs(y_test, y_pred, results_dir)

    print(f"Saved model: {model_path}")
    print(f"Saved training history: {history_path}")
    print(f"Saved loss curve: {loss_path}")
    print(f"Saved accuracy curve: {accuracy_path}")
    print(f"Saved classification report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()
