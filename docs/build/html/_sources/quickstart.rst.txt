Quickstart
==========

This quickstart guide introduces the core usage of the :mod:`MiniNumPy` package.
All examples reflect the actual behavior of the implementation.

Importing MiniNumPy
-------------------

To begin, import the package as:

.. code-block:: python

    import MiniNumPy as mnp

This provides access to NumPy-like functions such as ``mnp.array``, ``mnp.zeros``,
``mnp.arange``, and also the ``Array`` class internally used for computations.

Creating Arrays
---------------

MiniNumPy supports manual array creation using nested Python lists:

.. code-block:: python

    import MiniNumPy as mnp

    A = mnp.array([[1, 2],
                   [3, 4]])

    print(A)
    # [[1 2]
    #  [3 4]]

Array Creation Utilities
------------------------

MiniNumPy includes several helpers for creating common array types.

Zeros:

.. code-block:: python

    mnp.zeros((2, 3))
    # [[0 0 0]
    #  [0 0 0]]

Ones:

.. code-block:: python

    mnp.ones((2, 2))
    # [[1 1]
    #  [1 1]]

Identity matrix (``identity`` or ``eye``):

.. code-block:: python

    mnp.identity(3)
    # [[1 0 0]
    #  [0 1 0]
    #  [0 0 1]]

    mnp.eye(3, offset=1)
    # [[0 1 0]
    #  [0 0 1]
    #  [0 0 0]]

Ranges:

.. code-block:: python

    mnp.arange(5)
    # [0 1 2 3 4]

    mnp.arange(1, 6, 2)
    # [1 3 5]

Evenly spaced values:

.. code-block:: python

    mnp.linspace(0, 1, num=5)
    # [0.0 0.25 0.5 0.75 1.0]

Inspecting Array Shape and Size
-------------------------------

The :class:`Array` object exposes:

- ``shape`` — tuple of dimensions
- ``ndim`` — number of dimensions
- ``size`` — total number of elements

Example:

.. code-block:: python

    A = mnp.array([[1, 2],
                   [3, 4]])

    print(A.shape)   # (2, 2)
    print(A.ndim)    # 2
    print(A.size)    # 4

Reshaping and Transposing
-------------------------

Reshape:

.. code-block:: python

    B = A.reshape((4,))
    print(B)
    # [1 2 3 4]

Transpose:

.. code-block:: python

    print(A.T)
    # [[1 3]
    #  [2 4]]

Matrix multiplication is not in this section; it will appear in the Linear Algebra part of the quickstart.

Elementwise Operations
----------------------

MiniNumPy implements operator overloading for common arithmetic operations.

Addition:

.. code-block:: python

    X = mnp.array([1, 2, 3])
    Y = mnp.array([4, 5, 6])

    print(X + Y)
    # [5 7 9]

Subtraction:

.. code-block:: python

    print(X - Y)
    # [-3 -3 -3]

Scalar multiplication:

.. code-block::python

    print(X * 3)
    # [3 6 9]

Division:

.. code-block:: python

    print(X / 2)
    # [0.5 1.0 1.5]

Exponentiation:

.. code-block:: python

    print(X ** 2)
    # [1 4 9]

Elementwise Functions
---------------------

The :class:`Array` class provides:

- ``exp()``
- ``log()``
- ``sqrt()``
- ``abs()``

Example:

.. code-block:: python

    A = mnp.array([1, 4, 9])

    print(A.sqrt())
    # [1.0 2.0 3.0]

    print(A.log())
    # [0.0 1.386... 2.197...]

    print(A.exp())
    # [2.718... 54.598... 8103.083...]

Reductions
----------

MiniNumPy supports:

- ``sum()``
- ``mean()``
- ``max()``
- ``min()``
- ``arg_max()``
- ``arg_min()``

Example:

.. code-block:: python

    A = mnp.array([[3, 1],
                   [7, 5]])

    print(A.sum())      # 16.0
    print(A.mean())     # 4.0
    print(A.max())      # 7.0
    print(A.min())      # 1.0
    print(A.arg_max())  # (1, 0)
    print(A.arg_min())  # (0, 1)

===========================
Linear Algebra in MiniNumPy
===========================

MiniNumPy provides a compact set of linear algebra operations implemented using:

- LU decomposition
- Gauss–Jordan elimination
- Gram–Schmidt QR decomposition
- QR iteration for eigenvalues

All functions follow NumPy-like usage:

.. code-block:: python

    import MiniNumPy as mnp
    mnp.linalg.det(...)
    mnp.linalg.inv(...)
    mnp.linalg.eig(...)

Dot Product
-----------

The ``dot`` function supports:

- vector ⋅ vector → scalar
- column ⋅ column → scalar
- matrix @ matrix → matrix

Examples:

.. code-block:: python

    import MiniNumPy as mnp

    a = mnp.array([1, 2, 3])
    b = mnp.array([4, 5, 6])

    print(mnp.linalg.dot(a, b))
    # 32

    A = mnp.array([[1, 2],
                   [3, 4]])

    B = mnp.array([[5, 6],
                   [7, 8]])

    print(mnp.linalg.dot(A, B))
    # [[19 22]
    #  [43 50]]

Matrix Multiplication (matmul)
------------------------------

General matrix multiplication, supporting batch broadcasting.

.. code-block:: python

    A = mnp.array([[1, 0],
                   [0, 1]])

    B = mnp.array([[3, 4],
                   [5, 6]])

    print(mnp.linalg.matmul(A, B))
    # [[3 4]
    #  [5 6]]

Vector and Matrix Norms
-----------------------

Supported vector norms:

- ``ord=None`` or 2 → Euclidean norm
- ``ord=1`` → 1-norm
- ``ord=3`` → max norm

.. code-block:: python

    v = mnp.array([3, 4])

    print(mnp.linalg.norm(v))       # 5.0
    print(mnp.linalg.norm(v, 1))    # 7.0
    print(mnp.linalg.norm(v, 3))    # 4.0

Supported matrix norms:

- ``ord=1`` → max column sum
- ``ord=3`` → max row sum
- ``ord=2`` → spectral norm (via eigenvalues)

.. code-block:: python

    A = mnp.array([[1, -2],
                   [3,  4]])

    print(mnp.linalg.norm(A, 1))  # max column sum
    print(mnp.linalg.norm(A, 3))  # max row sum
    print(mnp.linalg.norm(A, 2))  # spectral norm

Determinant
-----------

Computed using LU decomposition:

.. code-block:: python

    A = mnp.array([[1, 2],
                   [3, 4]])

    print(mnp.linalg.det(A))
    # -2.0

Matrix Inversion
----------------

Computed using Gauss–Jordan elimination:

.. code-block:: python

    A = mnp.array([[1, 2],
                   [3, 4]])

    print(mnp.linalg.inv(A))
    # [[-2.   1. ]
    #  [ 1.5 -0.5]]

Solving Linear Systems
----------------------

Solve ``A x = b`` using LU decomposition:

.. code-block:: python

    A = mnp.array([[2, 1],
                   [5, 7]])

    b = mnp.array([11, 13])

    x = mnp.linalg.solve(A, b)
    print(x)
    # [[7.111...]
    #  [-3.222...]]

Eigenvalues and Eigenvectors
----------------------------

Computed using QR iteration + solving homogeneous systems:

.. code-block:: python

    B = mnp.array([[4, 1],
                   [1, 1]])

    vals, vecs = mnp.linalg.eig(B)

    print(vals)   # eigenvalues
    print(vecs)   # eigenvectors (columns)
