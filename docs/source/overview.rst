Overview
========

MiniNumPy is a lightweight educational re-implementation of selected NumPy
functionalities.  
This project is developed as part of the *Programming Course — M1 International
Track in Electrical and Engineering* and aims to provide a deeper
understanding of multidimensional arrays, broadcasting logic, elementwise
operations, and basic linear algebra routines.

The goal is **not** to replicate the full NumPy library, but to design a
minimal, structured, and functional subset that demonstrates how array
computation libraries work internally.

Purpose
-------

According to the project specification (see project PDF), MiniNumPy must:

- Implement a custom multidimensional array structure.
- Provide creation utilities such as ``array()``, ``zeros()``, ``ones()``,
  ``eye()``, ``arange()``, and ``linspace()``.
- Support elementwise arithmetic operations and reduction operations.
- Implement key linear algebra routines such as ``dot``, ``matmul``,
  ``norm``, ``det``, ``inv``, and optionally eigenvalue computation.
- Offer a simplified API that resembles NumPy:

  .. code-block:: python

      import MiniNumPy as mnp
      A = mnp.array([[1, 2], [3, 4]])
      val, vec = mnp.linalg.eig(A)

Project Structure
-----------------

The MiniNumPy library, following the specification, is composed of four main
components:

1. **Core Array Class**  
   Defines the internal representation of a multidimensional array, including:
   - data storage using nested Python lists  
   - attributes: ``shape``, ``ndim``, ``size``  
   - reshaping and transposition  
   - operator overloading for elementwise ops  

2. **Array Creation Functions**  
   Utilities for constructing arrays directly from Python data or numerical
   patterns:
   - ``mnp.array``  
   - ``mnp.zeros``  
   - ``mnp.ones``  
   - ``mnp.eye``  
   - ``mnp.arange``  
   - ``mnp.linspace``  

3. **Operations Module**  
   Elementwise arithmetic and reduction functions, including:
   - addition, subtraction, multiplication, division, exponentiation  
   - ``sum``, ``mean``, ``min``, ``max``  
   - broadcasting rules (simplified)  

4. **Linear Algebra Module**  
   Implements common linear algebra operations:
   - dot product and general matrix multiplication  
   - vector and matrix norms  
   - determinant and matrix inverse  
   - eigenvalue and eigenvector computation (bonus)  

Usage Philosophy
----------------

MiniNumPy is designed to mimic the familiar NumPy interface:

.. code-block:: python

    import MiniNumPy as mnp

    a = mnp.array([[1, 2],
                   [3, 4]])

    b = mnp.ones((2, 2))
    c = a + b
    d = mnp.linalg.inv(a)

This makes it possible to experiment with numerical computations without using
the full NumPy library, while understanding how such operations are implemented
internally.

Mini-Projects
-------------

The project PDF also specifies several small applications to validate the
library:

- **Image manipulation** (filtering, grayscale normalization, geometric
  transforms)
- **2D geometric operations** (rotation, scaling, shearing)
- **Matrix-based computation tasks**

These applications help connect the low-level implementation with real
engineering problems.

Summary
-------

MiniNumPy is both a learning tool and a minimal scientific computing library.
It emphasizes:

- clean API design
- algorithmic understanding  
- internal representation of arrays  
- implementation of numerical routines from first principles  

The rest of this documentation covers:

- Quickstart examples  
- Full API reference for each module  
- Technical notes explaining algorithms such as determinant, inverse, and
  eigenvalue computation  
