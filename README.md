# Least Squares Polynomial Curve Fitting

This project demonstrates polynomial curve fitting using the **Least Squares Method** and the **Normal Equations** approach.

## Overview

Given a set of data points, the objective is to find a polynomial that minimizes the sum of squared errors between the observed values and the predicted values.

The fitted polynomial has the form:

$$
P(x)=a_0+a_1x+a_2x^2+\cdots+a_nx^n
$$

The coefficients are obtained by constructing and solving the corresponding normal equation system.

## Features

* Polynomial fitting of arbitrary degree
* Least Squares implementation from scratch
* Normal Equations formulation
* Linear, Quadratic, and Cubic fitting examples
* Visualization of datasets and fitted curves
* Error evaluation for fitted models

## Project Structure

```text
.
├── find_coffecient.py          # Least Squares polynomial fitting implementation
├── least_common_square.ipynb   # Jupyter Notebook demonstration
└── README.md
```

## Mathematical Background

The algorithm minimizes:

$$
S=\sum_{i=1}^{m}(y_i-P(x_i))^2
$$

which leads to the normal equation system:

$$
A\mathbf{a}=\mathbf{d}
$$

where:

* $$(\mathbf{A})$$ is the coefficient matrix
* $$(\mathbf{a})$$ is the polynomial coefficient vector
* $$(\mathbf{d})$$ is the right-hand side vector

The system is solved using NumPy's linear algebra routines.

## Requirements

```bash
pip install numpy matplotlib
```

## Example

```python
from find_coffecient import find_coff_poly

data = [
    (1, 2.1),
    (2, 3.9),
    (3, 6.2),
    (4, 7.8),
    (5, 10.1)
]

coefficients = find_coff_poly(2, data)

print(coefficients)
```

## Applications

* Data approximation
* Trend analysis
* Scientific computing
* Engineering data modeling
* Numerical methods education

## Author

Developed as an educational project for understanding Least Squares Approximation and Polynomial Regression using Normal Equations.
