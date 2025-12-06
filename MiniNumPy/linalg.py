from MiniNumPy.Array import * 
import random

def dot(a: Array, b: Array):
    """ Dot product of two arrays.

    Args:
        a (Array): First input array.
        b (Array): Second input array.

    Returns:
        int : if both inputs are 1-D arrays, returns the dot product as a scalar.
        Array : if either input is a multi-dimensional array, returns the matrix product.
    """
    dot = 0
    if a.ndim == 1 and b.ndim ==1 and a.shape[0] == b.shape[0]:
        for i in range(a.size):
            dot += a.data[i]*b.data[i]
        return dot 
    
    #TODO: check
    if a.ndim ==2 and b.ndim ==2 and a.shape[1] == 1 and b.shape[1] == 1  :
        for i in range(a.size):
            dot += a.data[i][0]*b.data[i][0]
        return dot 
    
    else:
        return a@b

def matmul(A: Array, B: Array)-> Array:
    """ General matrix multiplication of two arrays A and B with broadcasting support.

    Args:
        A (Array): First input array.
        B (Array): Second input array.

    Raises:
        ValueError: If the last dimensions of A and B are incompatible for matrix multiplication.
        ValueError: If the batch dimensions of A and B cannot be broadcast together.

    Returns:
        Array: The result of the matrix multiplication with broadcasting applied.
    """
    if A.ndim < 2 or B.ndim < 2:
        raise ValueError("matmul: inputs must be at least 2-D arrays")
    
    # Compatible checking
    if A.shape[-1] != B.shape[-2]:
        raise ValueError("matmul: last dims mismatch")

    m = A.shape[-2]
    k = A.shape[-1]
    n = B.shape[-1]

    #Boardcast batch shape (take all dim except the last 2)
    batchA = A.shape[:-2]
    batchB = B.shape[:-2]

    batch_shape = broadcast_shapes(batchA, batchB)
    if batch_shape is None:
        raise ValueError("Cannot broadcast batch dims")

    #build the final shape
    out_shape = batch_shape + (m, n)

    flatA = A.flatten()
    flatB = B.flatten()
    out_flat = [0] * prod(out_shape)

    # For each output index, compute C[coord]
    for c_index in range(len(out_flat)):

        # coordinate of output element
        coordC = index_to_coord(c_index, out_shape)

        # split into: batch..., i, j
        batch = coordC[:-2]
        i = coordC[-2]
        j = coordC[-1]
        
        # From C find the A coord and Bcoord that construct C
        # map batch index to A index
        idxA = []
        off = len(batch_shape) - len(batchA) # off = 0 => A dim = out dim 
        for t in range(len(batchA)):
            dimA = batchA[t]
            dimOut = batch_shape[t + off]
            if dimA == dimOut:
                idxA.append(batch[t + off])
            else:
                idxA.append(0)

        # map batch index to B index
        idxB = []
        off = len(batch_shape) - len(batchB)
        for t in range(len(batchB)):
            dimB = batchB[t]
            dimOut = batch_shape[t + off]
            if dimB == dimOut:
                idxB.append(batch[t + off])
            else:
                idxB.append(0)

        # full coords for A and B inside batch
        # A: (...batch..., i, k)
        # B: (...batch..., k, j)
        rel = 0
        for kk in range(k):
            coordA = tuple(idxA + [i, kk])
            coordB = tuple(idxB + [kk, j])

            Ai = flatA[coord_to_index(coordA, A.shape)]
            Bj = flatB[coord_to_index(coordB, B.shape)]

            rel += Ai * Bj

        out_flat[c_index] = rel

    nested, _ = build_nested_list(out_flat, out_shape)
    # Total complexity: O(size)
    return Array(nested)

def broadcast_shapes(shapeA: tuple, shapeB: tuple)-> tuple:
    """Help function to broadcast two shapes.

    Args:
        shapeA (tuple): Shape of first array.
        shapeB (tuple): Shape of second array.

    Returns:
        tuple: The broadcasted shape, or None if they cannot be broadcast together.
    """
    # Check from right to left (from the closest level with last 2 dim to the most furthur)
    a = list(shapeA)[::-1]
    b = list(shapeB)[::-1]
    out = []
    # build the boardcasting shape
    for i in range (max(len(a),len(b))):
        dimA = a[i] if i < len(a) else 1
        dimB = b[i] if i < len(b) else 1
        
        if dimA == dimB:
            out.append(dimA)
        elif dimA == 1:
            out.append(dimB)
        elif dimB == 1:
            out.append(dimA)
        else:
            return None
    return tuple(out[::-1])

def norm(a:Array, ord = None)->float:
    """Calculate the norm of an array.

    Args:
        a (Array): Input Array.
        ord (int, optional): The norm order. Defaults to 2.

    Raises:
        ValueError: If the input is not of Array type.

    Returns:
        float: The calculated norm of the array.
    """
    if not isinstance(a,Array):
        raise ValueError("The input must be in Array type")
    
    if a.ndim !=2:
        a_flat = a.flatten()
        vals = [abs(x) for x in a_flat]
        
        if ord == None or ord == 2:
            norm = 0
            for i in a_flat:
                norm += i**2
            return float(math.sqrt(norm))
        
        if ord == 1:
            norm = 0
            for i in vals:
                norm += i
            return float(norm)
        
        if ord == 3:
            return float(max(vals))
    
    if ord == None:
        a_flat = a.flatten()
        norm = 0
        for i in a_flat:
            norm += i**2
        return float(math.sqrt(norm))    
    
    # Max column sum 
    if ord == 1:
        cols_sum = []
        for j in range(a.shape[1]):
            s = 0
            cols = a.get_col(j)
            for i in range(a.shape[0]):
                s += abs(cols.data[i][0])
            cols_sum.append(s)
        return float(max(cols_sum))
    
    # Max sigma (sqrt of eigein value of ATA)
    if ord ==2:
        ATA = a.T@a
        eigenvalue,_ = eig(ATA)
        return eigenvalue.max()**0.5
    
    # Max row sum 
    if ord == 3:
        rows_sum = []
        for i in range(a.shape[0]):
            s = 0 
            rows = a.get_row(i)
            for j in range(a.shape[1]):
                s+= abs(rows.data[j])
            rows_sum.append(s)
        return float(max(rows_sum))
    
def det(a:Array):
    return a.determinant()

def solve(A: Array, B: Array) -> Array:
    """
    Solve the linear system A X = B where:
      - A is an nxn square matrix
      - B is either:
            (n,)   → 1-D vector
            (n,1)  → column vector
            (n,k)  → multiple right-hand sides

    Returns:
      - X with the same second dimension as B (nx1 or nxk)
    """

    # A must be an n×n square matrix
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix")

    n = A.shape[0]
    
    try:
            P, L, U = A.LU_Decomposition()
    except ValueError:
       return "System has infinitely many solutions or no solution"
    
    # Convert B into a proper 2-D matrix form (column vectors)
    if B.ndim == 1:
        # Convert shape (n,) -> (n,1)
        Bmat = Array([[B.data[i]] for i in range(n)])
    elif B.ndim == 2 and B.shape[0] == n:
        # Already (n,1) or (n,k)
        Bmat = B.copy()
    else:
        raise ValueError("B must have shape (n,), (n,1), or (n,k)")
    
    Bmat = P @ Bmat
        
    # Number of columns of B (number of RHS vectors)
    k = Bmat.shape[1]

    # Forward substitution: solve L Y = B
    Y = zeros((n, k))
    for col in range(k):
        for i in range(n):
            s = 0
            for j in range(i):
                s += L.data[i][j] * Y.data[j][col]
            Y.data[i][col] = Bmat.data[i][col] - s

    # Backward substitution: solve U X = Y
    X = zeros((n, k))
    for col in range(k):
        for i in range(n - 1, -1, -1):
            s = 0
            for j in range(i + 1, n):
                s += U.data[i][j] * X.data[j][col]
            X.data[i][col] = (Y.data[i][col] - s) / U.data[i][i]

    return X 

def inv(a:Array)->Array:
    """Calculate the inverse of a square matrix using Gauss-Jordan elimination.

    Args:
        a (Array): Input square matrix.

    Raises:
        ValueError: If the input matrix is not square.
        ValueError: If the matrix is singular and cannot be inverted.

    Returns:
        Array: The inverse of the input matrix.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Only square 2D arrays can be inverted")
    
    n = a.shape[0]

    I = identity(n)
    A = a.copy()
    
    for p in range (n):
        
        if A.data[p][p] == 0:
            for r in range(p + 1, n):
                if A.data[r][p] != 0:
                    A.data[p], A.data[r] = A.data[r], A.data[p]
                    I.data[p], I.data[r] = I.data[r], I.data[p]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted")
            
        pivot = A.data[p][p]
        
        for j in range(n):
            A.data[p][j] /= pivot
            I.data[p][j] /= pivot
        
        for i in range(n):
            if i != p:
                factor = A.data[i][p]
                for j in range(n):
                    A.data[i][j] -= factor *A.data[p][j]
                    I.data[i][j] -= factor *I.data[p][j]
    
    return I

#TODO: Eigenvalue using QRD

def qr_decomposition(A: Array):
    """
    Calculate the QR decomposition of matrix A using the Gram-Schmidt process.
        Idea: A = Q R
        where:
            Q: orthogonal matrix (columns are orthonormal vectors)
            R: upper triangular matrix
            
    Args:
        A (Array): Input matrix to decompose.

    Returns:
        Q (Array) : Orthogonal matrix.
        R (Array) : Upper triangular matrix.
    """
    m, n = A.shape

    Q = zeros((m, n))
    R = zeros((n, n))

    for j in range(n):
        v = A.get_col(j) # m x 1

        for i in range(j):
            qi = Q.get_col(i) # m x 1
            R.data[i][j] = dot(qi,v) # v = aj 
            v = v - qi * R.data[i][j]

        norm_v = norm(v)
        R.data[j][j] = norm_v
        qj = v / norm_v

        Q.set_col(j, qj)

    return Q, R

def eig(B:Array, max_iter=200, eps=1e-6):
    """Calculate the eigenvalues and eigenvectors of a square matrix using the QR algorithm.

    Args:
        B (Array): Input square matrix.
        max_iter (int, optional): Maximum number of iteration. Defaults to 200.
        eps (float, optional): Convergence thresshold. Defaults to 1e-6.

    Raises:
        ValueError: If the input matrix is not square.

    Returns:
        eigenvalues (Array): The eigenvalues of the matrix.
        V (Array): The eigenvectors of the matrix as columns.
    """
    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("Eigenvalue computation only supports square 2D arrays")
    
    A = B.copy()
    n = A.shape[0]

    # eigenvector accumulator
    V = identity(n)

    for _ in range(max_iter):
        Q, R = qr_decomposition(A)

        A = R @ Q      # QR iteration
        V @= Q      # accumulate eigenvectors

        # check convergence
        converged = True
        for i in range(n):
            for j in range(i):
                if abs(A.data[i][j]) > eps:
                    converged = False
                    break # if one element is not converged, skip the rest
            if not converged:
                break # if not converged, skip the rest, continue to next iteration

        if converged:
            break

    # eigenvalues = diagonal
    eigenvalues = [A.data[i][i] for i in range(n)]

    # for j in range(n):
    #     Vcol= V.get_col(j)
    #     Vcol /= Vcol.data[n-1][0]
    #     V.set_col(j,Vcol)
        
    return Array(eigenvalues), V

