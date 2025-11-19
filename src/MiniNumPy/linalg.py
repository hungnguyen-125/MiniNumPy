from MiniNumPy.Array import * 

def norm(a, ord = 2)->float:
    if not isinstance(a,Array):
        raise ValueError("The input must be in Array type")
    
    a_flat = a.flatten()
    norm = 0
    for i in a_flat:
        norm += i**ord
    return float(norm)

def det(a:Array):
    return a.determinant()

def solve(a: Array, b: Array ) -> Array:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Solve function just can be applied on 2-D square matrix')
    if a.shape[1] != b.shape[0] or b.ndim != 1:
        raise ValueError('Target vector is invalid')
    
    n = a.shape[0]
    
    L, U = a.LU_Decomposition()
    x_tide = [0]*n
    for i in range(n):
        s = 0
        for j in range(i):
            s += L.data[i][j]*x_tide[j]
        x_tide[i] = b.data[i] - s
    
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += U.data[i][j]*x[j]
        x[i] = (x_tide[i] - s)/U.data[i][i]
    
    return Array(x)

#TODO: Practice more on using pivot
def inv(a:Array):
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Only square 2D arrays can be inverted")
    
    n = a.shape[0]
    # Create an identity matrix of the same size
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    # Create a copy of the original matrix
    A = a.copy()
    
    for p in range (n):
        pivot = A.data[p][p]
        if pivot == 0:
            raise ValueError('Matrix is singular and cant be inverted')
        
        for j in range(n):
            A.data[p][j] /= pivot
            I[p][j] /= pivot
        
        for i in range(n):
            if i != p:
                factor = A.data[i][p]
                for j in range(n):
                    A.data[i][j] -= factor *A.data[p][j]
                    I[i][j] -= factor *I[p][j]
                    I[i][j] = round(I[i][j], 8)
                    
    for j in range(n):
        I[n-1][j] = round(I[n-1][j], 8)
    
    return Array(I)

#TODO: Eigenvalue using QR decomposition => eigenvector
def eig(a: Array):
    pass