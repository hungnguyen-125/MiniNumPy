Array Module
============

The ``MiniNumPy`` package provides the fundamental ``array`` constructor and
the underlying ``Array`` class, forming the core of its computational model.

.. code-block:: python

    import MiniNumPy as mnp
    a = mnp.array([[1, 2], [3, 4]])

Overview
--------

The ``array`` constructor creates a MiniNumPy Array object that supports:

- Arbitrary-dimensional arrays stored as nested Python lists.
- Attributes: ``shape``, ``ndim``, ``size``, ``data``.
- Reshape and transpose operations.
- Elementwise arithmetic (``+``, ``-``, ``*``, ``/``, ``**``).
- Matrix multiplication (``@``).

==================
Array Class API
==================

.. currentmodule:: MiniNumPy.Array

.. autosummary::
   :toctree: _autosummary/array_class
   :caption: Array Class Methods
   :nosignatures:

   Array

.. autoclass:: MiniNumPy.Array.Array
   :members:
   :undoc-members:
   :special-members: __str__, __add__, __sub__, __mul__, __truediv__, __pow__, __matmul__
   :show-inheritance:

====================
Creation Functions
====================

.. autosummary::
   :toctree: _autosummary/creation
   :caption: Creation Functions
   :nosignatures:

   zeros
   ones
   identity
   eye
   arange
   linspace

==================
Helper Functions
==================

.. autosummary::
   :toctree: _autosummary/helper
   :caption: Helper Functions
   :nosignatures:

   build_nested_list
   index_to_coord
   coord_to_index
   
Examples
--------

.. code-block:: python

    import MiniNumPy as mnp

    A = mnp.array([[1, 2],
                   [3, 4]])

    print(A.T)
    # [[1 3]
    #  [2 4]]

    print(A + mnp.ones((2, 2)))
    # [[2 3]
    #  [4 5]]
