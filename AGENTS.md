# Repository Instructions

This is a beginner-friendly capstone project for MNIST handwritten digit classification with a CNN.

## Workflow

- Run commands from the repository root.
- Use `python3` on macOS/Linux unless a virtual environment provides `python`.
- Train with `python3 src/train.py`.
- Predict with `python3 src/predict.py --image sample_images/my_digit.png`.
- Keep the MNIST test set separate for final evaluation only.

## Generated Files

The training script writes generated artifacts to `models/` and `results/`. These artifacts are ignored by Git by default because they can be recreated.

## Editing Notes

- Keep the notebook and scripts aligned when changing model architecture or preprocessing.
- Avoid replacing capstone notebooks or report files without making a backup first.
