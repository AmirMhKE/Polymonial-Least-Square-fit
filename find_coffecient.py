import numpy as np

def find_coff_poly(n, data):
    """
    Compute the coefficients of the best-fit polynomial of degree n using
    the Least Squares Normal Equations method.

    This function constructs and solves the normal equation system A * x = d,
    where A is an (n+1) x (n+1) matrix and d is a vector of length (n+1).

    Parameters
    ----------
    n : int
        Degree of the polynomial to fit.
        - n=1 : linear      (a0 + a1*x)
        - n=2 : quadratic   (a0 + a1*x + a2*x^2)
        - n=3 : cubic       (a0 + a1*x + a2*x^2 + a3*x^3)

    data : list of tuples
        A list of (x, y) data points to fit.
        Example: [(1, 2.1), (2, 3.9), (3, 6.2)]

    Returns
    -------
    x : numpy.ndarray, shape (n+1,)
        Polynomial coefficients ordered from lowest to highest degree:
        [a0, a1, a2, ..., an]

        The fitted polynomial is:
            P(x) = a0 + a1*x + a2*x^2 + ... + an*x^n

    Raises
    ------
    numpy.linalg.LinAlgError
        If the normal equation matrix A is singular (e.g., duplicate x values
        or insufficient data points for the chosen degree).

    Notes
    -----
    - Requires at least (n+1) data points.
    - The normal equation matrix A[i][j] = Σ x_k^(i+j) may become
      ill-conditioned for high-degree polynomials (n > 6).
      In such cases, consider using numpy.polyfit() instead.

    Examples
    --------
    >>> data = [(1, 3.6), (2, 7.1), (3, 11.8), (4, 17.2), (5, 23.9)]
    >>> coeffs = find_coff_poly(2, data)
    >>> print(coeffs)
    [a0, a1, a2]   # P(x) = a0 + a1*x + a2*x^2
    """
    # Equation coefficients
    A = []
    # The values ​​on the right side of the equations
    d = []
    # Number of points
    m = len(data)
    for i in range(0, n + 1):
        row = []
        for j in range(0, n + 1):
            coff = 0
            for k in range(m):
                coff += data[k][0] ** (j + i)
            row.append(coff)
        A.append(row)

    for i in range(0, n + 1):
        value = 0
        for j in range(m):
            value += (data[j][0] ** i) * data[j][1]
        d.append(value)

    A = np.array(A, dtype=float)
    d = np.array(d, dtype=float)
    # Solving the system of equations
    # a0, a1, ..., an
    # an.x^n + an-1.x^n-1 + ... + a0
    x = np.linalg.solve(A, d)
    return x
