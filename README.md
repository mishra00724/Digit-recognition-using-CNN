# MNIST Handwritten Digit Classification using CNN

A deep learning project that trains a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset and allows prediction on custom digit images.

This project was developed as a capstone-style machine learning project using TensorFlow/Keras.

---

## Project Overview

The objective of this project is to build an image classification model that can recognize handwritten digits from **0 to 9**.

The model is trained on the **MNIST dataset**, which contains grayscale handwritten digit images. After training, the model is exported in Keras format and can be reused to predict digits from external image files.

---

## Problem Statement

Handwritten digit recognition is a classic computer vision problem where the goal is to classify an input image of a handwritten digit into one of ten possible classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

This project solves the problem by training a Convolutional Neural Network that automatically learns visual patterns such as edges, curves, loops, and strokes from digit images.

---

## Key Features

- Train a CNN model on the MNIST dataset
- Evaluate model performance on a separate test set
- Save/export the trained model in `.keras` format
- Predict custom handwritten digit images
- Generate training history, accuracy/loss plots, classification report, and confusion matrix
- Includes a Jupyter notebook and project report for capstone submission

---

## Repository Structure

```text
mnist-cnn-digit-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── capstone_mnist_cnn_submission_notebook.ipynb
│
├── reports/
│   └── .gitkeep
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── sample_images/
│   └── .gitkeep
│
└── assets/
    └── .gitkeep
```

---

## Dataset Description

This project uses the **MNIST handwritten digit dataset**, available directly through TensorFlow/Keras.

| Item | Description |
|---|---|
| Dataset | MNIST |
| Image Type | Grayscale |
| Image Size | 28 x 28 pixels |
| Number of Classes | 10 |
| Classes | Digits 0 to 9 |
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Source | TensorFlow/Keras datasets API |

The dataset is loaded using:

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

---

## Model Architecture

The model uses a Convolutional Neural Network architecture.

| Layer | Configuration | Purpose |
|---|---|---|
| Conv2D | 32 filters, 4x4 kernel, ReLU | Extract digit features such as edges, curves, and strokes |
| MaxPool2D | 2x2 pooling | Reduce feature map size and retain important features |
| Flatten | - | Convert feature maps into a one-dimensional vector |
| Dense | 128 neurons, ReLU | Learn high-level digit patterns |
| Dropout | 0.3 | Reduce overfitting |
| Dense Output | 10 neurons, Softmax | Output class probabilities for digits 0 to 9 |

The output layer uses **Softmax** activation because this is a multiclass classification problem.

---

## Evaluation Metrics

The model is evaluated using:

- Training accuracy
- Validation accuracy
- Test accuracy
- Training loss
- Validation loss
- Test loss
- Classification report
- Confusion matrix

The loss function used is:

```text
categorical_crossentropy
```

The optimizer used is:

```text
Adam
```

---

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mnist-cnn-digit-classifier.git
cd mnist-cnn-digit-classifier
```

Replace `your-username` with your actual GitHub username.

---

### 2. Create a Virtual Environment

#### macOS/Linux

```bash
python3 -m venv minst_env
source minst_env/bin/activate
```

#### Windows

```bash
python -m venv minst_env
minst_env\Scripts\activate
```

---

### 3. Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

---

## Requirements

The repository includes a `requirements.txt` file with the following dependencies:

```txt
tensorflow>=2.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pillow>=10.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

TensorFlow support depends on your Python version. If installation fails on a very new Python release, create a virtual environment with a TensorFlow-supported Python version such as Python 3.10, 3.11, or 3.12.

---

## How to Train the Model

Run the training script:

```bash
python3 src/train.py
```

For a quick smoke run, reduce the number of epochs:

```bash
python3 src/train.py --epochs 1 --batch-size 128
```

The script will:

1. Load the MNIST dataset
2. Preprocess and normalize images
3. One-hot encode labels
4. Create a train-validation split
5. Build and train the CNN model
6. Evaluate the model on the test set
7. Save the trained model
8. Save training plots and evaluation outputs

---

## Training Outputs

After training, the following files will be generated.

### Saved Model

```text
models/mnist_cnn_model.keras
```

### Results

```text
results/training_history.csv
results/loss_curve.png
results/accuracy_curve.png
results/classification_report.txt
results/confusion_matrix.png
```

---

## How to Predict a Digit from an Image

After training, place your image inside the `sample_images/` folder.

Example:

```text
sample_images/my_digit.png
```

Then run:

```bash
python3 src/predict.py --image sample_images/my_digit.png
```

If your image has a **black digit on a white background**, run:

```bash
python3 src/predict.py --image sample_images/my_digit.png --invert
```

Example output:

```text
Predicted Digit: 7
Confidence: 0.9821
```

---

## Input Image Guidelines

For best prediction results, use an image that has:

- One handwritten digit only
- A clear and centered digit
- Minimal background noise
- PNG or JPG format
- Black digit on white background, or white digit on black background

The prediction script automatically:

1. Converts the image to grayscale
2. Resizes the image to 28 x 28 pixels
3. Normalizes pixel values
4. Optionally inverts image colors
5. Reshapes the image for the CNN model
6. Predicts the digit class

---

## Example Prediction Command

```bash
python3 src/predict.py --image sample_images/digit_7.png --invert
```

Output:

```text
Loading model from: models/mnist_cnn_model.keras
Predicted Digit: 7
Confidence: 0.98
```

---

## Running the Jupyter Notebook

To launch Jupyter Notebook:

```bash
python -m ipykernel install --user --name minst_env --display-name "Python (minst_env)"
jupyter notebook
```

Open:

```text
notebooks/capstone_mnist_cnn_submission_notebook.ipynb
```

In Jupyter, select:

```text
Kernel -> Change Kernel -> Python (minst_env)
```

The notebook contains:

- Full implementation
- Dataset exploration
- Model training
- Model evaluation
- Model export
- External image prediction function
- Saved outputs for capstone verification

---

## Project Report

Place the capstone project report in:

```text
reports/capstone_mnist_cnn_project_report_updated.docx
```

The report should include:

- Problem statement
- Dataset description
- Methodology
- Neural network architecture
- Experimental details
- Evaluation metrics
- Results and discussion
- Improvement suggestions

---

## Model Saving and Loading

### Save Model

```python
model.save("models/mnist_cnn_model.keras")
```

### Load Model

```python
from tensorflow.keras.models import load_model

model = load_model("models/mnist_cnn_model.keras")
```

---

## Prediction Workflow

```text
Input Image
   ↓
Convert to Grayscale
   ↓
Resize to 28 x 28
   ↓
Normalize Pixel Values
   ↓
Optional Color Inversion
   ↓
Reshape to (1, 28, 28, 1)
   ↓
Load Saved CNN Model
   ↓
Predict Class Probabilities
   ↓
Return Predicted Digit
```

---

## Repository Health Check

Current workflow status:

- `src/train.py`, `src/predict.py`, and `src/utils.py` are runnable from the project root.
- Model files are saved under `models/`.
- Training history, plots, classification report, and confusion matrix are saved under `results/`.
- The notebook folder is present and contains the capstone notebook.
- `reports/` and `sample_images/` are present with placeholders; add the final report and custom digit images there.
- The MNIST test set is kept separate for final evaluation.
- Use `python3 src/train.py --help` and `python3 src/predict.py --help` to see available command-line options.

---

## Gitignore Policy

The repository includes a `.gitignore` that excludes local environments, cache files, local backups, and large trained model artifacts. Small result files are kept so the capstone outputs are visible in GitHub:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
venv/
env/
.venv/
minst_env/
home_testandrun/

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/

# Large files and data
data/

# Trained model artifacts
models/*.keras
models/*.h5
models/*.pb

# Local document backups
*.backup.docx
```

> Note: If you want to include the trained model in your GitHub repository, remove `models/*.keras` from `.gitignore`.

---

## Common Issues and Fixes

### 1. TensorFlow Installation Error

Try upgrading pip:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow
```

### 2. Model File Not Found

If you see an error like:

```text
No such file or directory: models/mnist_cnn_model.keras
```

Train the model first:

```bash
python3 src/train.py
```

### 3. Wrong Prediction on Custom Image

Try using the `--invert` flag:

```bash
python3 src/predict.py --image sample_images/my_digit.png --invert
```

Also make sure the digit is centered and clear.

### 4. Jupyter Notebook Does Not Open

Install Jupyter:

```bash
python3 -m pip install jupyter
```

Then run:

```bash
jupyter notebook
```

---

## Future Improvements

Possible improvements for this project:

1. Add more convolutional layers
2. Use Batch Normalization
3. Apply data augmentation carefully
4. Analyze misclassified images
5. Improve preprocessing for real-world handwritten images
6. Build a Streamlit or Flask web app
7. Train on Fashion-MNIST, EMNIST, or custom handwritten digit data
8. Add Docker support for reproducible deployment
9. Add unit tests for preprocessing and prediction functions

---

## Technologies Used

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter Notebook

---

## Author

**Himanshu Mishra**

---

## License

This project is created for educational and capstone submission purposes.

You may modify and extend it for learning, portfolio building, or further experimentation.
