Linear Algebra Module
=====================

The ``MiniNumPy.linalg`` submodule provides fundamental linear algebra
operations following a NumPy-like API:

.. code-block:: python

    import MiniNumPy as mnp
    mnp.linalg.eig(...)

Overview
--------

Available operations include:

- ``mnp.linalg.dot(a, b)`` — dot product
- ``mnp.linalg.matmul(a, b)`` — matrix multiplication (with broadcasting)
- ``mnp.linalg.norm(a)`` — vector/matrix norm
- ``mnp.linalg.det(a)`` — determinant
- ``mnp.linalg.inv(a)`` — matrix inverse
- ``mnp.linalg.solve(A, b)`` — solve linear system
- ``mnp.linalg.eig(a)`` — eigenvalues & eigenvectors
- ``mnp.linalg.qr_decomposition(a)`` — QR decomposition (internal)

API Reference
-------------

.. currentmodule:: MiniNumPy.linalg

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   dot
   matmul
   norm
   det
   solve
   inv
   qr_decomposition
   eig

.. automodule:: MiniNumPy.linalg
   :members:
   :undoc-members:
   :show-inheritance:
   :autosummary:

Examples
--------

Dot product:

.. code-block:: python

    import MiniNumPy as mnp

    a = mnp.array([1, 2, 3])
    b = mnp.array([4, 5, 6])

    print(mnp.linalg.dot(a, b))  # 32

Matrix multiplication:

.. code-block:: python

    A = mnp.array([[1, 0],
                   [0, 1]])
    B = mnp.array([[3, 4],
                   [5, 6]])

    print(mnp.linalg.matmul(A, B))

Determinant:

.. code-block:: python

    M = mnp.array([[1, 2],
                   [3, 4]])

    print(mnp.linalg.det(M))  # -2

Solve linear system:

.. code-block:: python

    A = mnp.array([[2, 1],
                   [5, 7]])
    b = mnp.array([11, 13])

    print(mnp.linalg.solve(A, b))

Eigenvalues & eigenvectors:

.. code-block:: python

    B = mnp.array([[4, 1],
                   [1, 1]])

    vals, vecs = mnp.linalg.eig(B)
    print(vals)
    print(vecs)
