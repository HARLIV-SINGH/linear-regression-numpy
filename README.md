# Linear Regression from Scratch (NumPy)

This repository contains an implementation of **linear regression** from scratch using only **NumPy**. It trains a regression model on the **California Housing dataset** using gradient descent, with no reliance on libraries like scikit-learn.

The goal is to practice the **mathematical foundations** of regression: preprocessing, normalization, gradient-based optimization, and evaluation metrics.

---

## Features

* Data preprocessing with:

  * One-hot encoding (`ocean_proximity`)
  * Median imputation for missing values
  * Train-test split (80/20)
  * Feature scaling (mean normalization + standard deviation scaling)
* Linear regression using **gradient descent** with early stopping.
* Evaluation with:

  * **Root Mean Squared Error (RMSE)**
  * **Mean Absolute Error (MAE)**
  * **R² score**
* Training loss visualization with matplotlib.

---

## Files

* `linear_regression.ipynb` - Notebook to understand step-by-step.
* `linear_regression.py` - Code to get the testing results directly.
* `housing.csv` – Dataset (California Housing).

---

## Installation

Clone the repo:

```bash
git clone https://github.com/HARLIV-SINGH/linear-regression-numpy.git
cd linear-regression-numpy
```

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

## Usage

### Option 1 – Run Jupyter Notebook

To explore the math and code step by step, launch the notebook:

```bash
jupyter notebook linear_regression.ipynb
```

This will open an interactive interface in your browser. You can run cells one by one to see the implementation and results.

### Option 2 – Run Python Script

If you just want to execute the core code:

```bash
python linear_regression.py
```

This will train the model and display results directly in the terminal.

The code:
* Trains the model on 80% of the dataset
* Evaluates on the remaining 20%
* Prints RMSE, MAE, and R² score
* Plots the training loss over epochs

---

## Example Output

```
Early stopping at epoch 1358
RMSE indicates the average magnitude of error between actual and predicted values
RMSE: 71594.15

MAE calculates the average of the absolute differences between actual and predicted values
MAE: 51000.83

R² score indicates how well the model fits the data
R² Score: 0.6178
```

Training loss curve:

<img width="843" height="578" alt="image" src="https://github.com/user-attachments/assets/8a95d2e7-f44f-4f20-8dc3-bd69565b9895" />

---

## Next Steps

* Extend to polynomial regression.
* Compare with models from standard libraries.
