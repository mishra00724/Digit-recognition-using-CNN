import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def build_model():
    """
    Build and return the CNN model for MNIST digit classification.
    """

    model = Sequential()

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(4, 4),
            activation="relu",
            input_shape=(28, 28, 1)
        )
    )

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))

    model.add(Dropout(0.3))

    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model


def main():
    """
    Train the CNN model on MNIST dataset and save outputs.
    """

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Training images shape:", x_train.shape)
    print("Testing images shape:", x_test.shape)

    # Reshape images for CNN input: (samples, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32")
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32")

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-hot encode labels
    y_cat_train = to_categorical(y_train, num_classes=10)
    y_cat_test = to_categorical(y_test, num_classes=10)

    # Create validation split from training data
    x_train_final, x_valid, y_train_final, y_valid = train_test_split(
        x_train,
        y_cat_train,
        test_size=0.2,
        random_state=22,
        stratify=y_train
    )

    print("Training split:", x_train_final.shape)
    print("Validation split:", x_valid.shape)
    print("Test split:", x_test.shape)

    model = build_model()

    print("Model architecture:")
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    print("Training model...")

    history = model.fit(
        x_train_final,
        y_train_final,
        epochs=10,
        batch_size=128,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stop]
    )

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("results/training_history.csv", index=False)

    # Plot loss curve
    history_df[["loss", "val_loss"]].plot(figsize=(8, 5))
    plt.title("Training Loss vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("results/loss_curve.png", bbox_inches="tight")
    plt.close()

    # Plot accuracy curve
    history_df[["accuracy", "val_accuracy"]].plot(figsize=(8, 5))
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("results/accuracy_curve.png", bbox_inches="tight")
    plt.close()

    print("Evaluating model on test set...")

    test_loss, test_accuracy = model.evaluate(x_test, y_cat_test)

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Generate predictions
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Save classification report
    report = classification_report(y_test, y_pred_classes)

    with open("results/classification_report.txt", "w") as file:
        file.write(report)

    print("Classification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig("results/confusion_matrix.png", bbox_inches="tight")
    plt.close()

    # Save trained model
    model.save("models/mnist_cnn_model.keras")

    print("Model saved at: models/mnist_cnn_model.keras")
    print("Training results saved in: results/")


if __name__ == "__main__":
    main()