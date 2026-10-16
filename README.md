# Support Vector Machine Fundamentals

A Python and NumPy implementation of core SVM calculations for exploring decision boundaries, margins, and margin violations on two-dimensional labeled data.

## Features

- **Weight vector calculation:** Computes the primal weight vector by summing each training example's contribution, weighted by its alpha value and class label.
- **Margin point identification:** Returns the indices of examples on their respective class margin boundaries, using a default tolerance of `0.001` for numerical imprecision.
- **Slack detection:** Returns the indices of examples that violate the margin constraint, including correctly classified points inside the margin and misclassified points.

These functions evaluate supplied alpha values or boundary parameters; training an SVM optimizer is outside the project's scope.

## Datasets

`svm.py` includes two small datasets, `kSEP` and `kINSP`. Each row contains two features followed by a class label of `+1` or `-1`.

## How it works

The weight vector is calculated as:

$$w = \sum_i \alpha_i y_i x_i$$

Each example receives a score of $w \cdot x_i + b$. The decision boundary has score zero, and the class margin boundaries have scores `+1` and `-1`.

`find_support` identifies examples whose scores are within the tolerance of their class labels. `find_slack` identifies examples for which $y_i(w \cdot x_i + b) < 1$.

## Usage

Requires Python and NumPy:

```bash
python -m pip install numpy
```

```python
from numpy import array
from svm import kSEP, find_support, find_slack

x = kSEP[:, :2]
y = kSEP[:, 2]
w = array([0.2, 0.8])
b = -0.2

print(find_support(x, y, w, b))  # Indices: {0, 2, 4}
print(find_slack(x, y, w, b))
```

## Validation

The implementation passed all four supplied unit tests, covering weight calculation, margin point identification, and two slack detection scenarios. The test file is excluded from this repository.
