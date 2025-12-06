from __future__ import annotations
import math

def build_nested_list(flat_data:list, shape:tuple, offset=0):
    """Build a nested list from a flat list based on the given shape.

    Args:
        flat_data (list): A flat list containing the data elements.
        shape (tuple): The desired shape of the nested list.
        offset (int, optional): Work as a pointer point to the data will take in next loop (tracking pointer). This is used internally during recursion. Defaults to 0.

    Returns:
        nested_list (list): A nested list structured according to the specified shape.
        offset (int): The updated offset after building the nested list.
    """

    # If shape empty => return a single element
    if len(shape) == 0:
        return flat_data[offset], offset + 1

    # allocate this level
    dim = shape[0]
    sub_shape = shape[1:]
    nested_list = []

    # recursively fill
    for _ in range(dim):
        item, offset = build_nested_list(flat_data, sub_shape, offset)
        nested_list.append(item)

    return nested_list, offset
        
    
def array(data):
    return Array(data)
    
def _fill(shape:tuple, fill_value:float)->list:
    """Helper function to create a nested list filled with a specific value.

    Args:
        shape (tuple): The shape of the nested list to create.
        fill_value (float): The value to fill the nested list with.

    Returns:
        list: The nested list filled with the specified value.
    """
    size = prod(shape)
    flat = [fill_value]*size
    filled_arr, _ = build_nested_list(flat,shape)
    return filled_arr

def zeros(shape:tuple)->Array:
    """Create an Array filled with zeros.

    Args:
        shape (tuple): Shape of the Array

    Returns:
        Array: An Array of the specified shape filled with zeros.
    """
    data = _fill(shape, 0)
    return Array(data)

def ones(shape:tuple)->Array:
    """Create an Array filled with ones.

    Args:
        shape (tuple): Shape of the Array

    Returns:
        Array: An Array of the specified shape filled with ones.
    """
    data = _fill(shape, 1)
    return Array(data)

def identity(n:int)->Array:
    """Create an identity matrix of size n x n.

    Args:
        n (int): Size of the identity matrix

    Returns:
        Array: An n x n identity matrix as an Array.
    """
    data = _fill((n, n), 0)
    for i in range(n):
        data[i][i] = 1
    return Array(data)

def eye(n: int, m = None, offset: int = 0)->Array:
    """Create a 2D Array with ones on a specified diagonal and zeros elsewhere.
        - If offset = 0 => main diagonal.
        - If offset > 0 => diagonal shifted to the right "offset" columns.
        - If offset < 0 => diagonal shifted downward "offset" rows.
    Args:
        n (int): Number of rows.
        m (int, optional): Number of column. Defaults to None.
        offset (int, optional): Index of the diagonal. Defaults to 0.

    Returns:
        Array: A 2D Array with ones on the specified diagonal and zeros elsewhere.
    """
    if m is None: 
        m = n
        
    data = _fill((n, m), 0)
    for i in range(n):
        j = i + offset
        if 0 <= j < m:
            data[i][j] = 1
    return Array(data)

def arange(start:float, stop=None, step=1)-> Array:
    """Create a 1D Array with evenly spaced values within a given interval.

    Args:
        start (float): Start of the interval.
        stop (float, optional): End of the interval. Defaults to None.
        step (float, optional): The spacing between values. Defaults to 1.

    Returns:
        Array: A 1D Array containing evenly spaced values.
    """
    if stop is None:
        stop = start
        start = 0
    data = []
    value = start
    while (step > 0 and value < stop) or (step < 0 and value > stop):
        data.append(value)
        value += step
    return Array(data)

def linspace(start:float, stop:float, num=50) -> Array:
    """Create a 1D Array with evenly spaced values over a specified interval.

    Args:
        start (float): Start of the interval.
        stop (float): End of the interval.
        num (int, optional): Number of elements . Defaults to 50.

    Returns:
        Array: A 1D Array containing num evenly spaced values from start to stop.
    """
    if num <= 0:
        return Array([])
    if num == 1:
        return Array([start])
    step = (stop - start) / (num - 1)
    data = [start + i * step for i in range(num)]
    return Array(data)

def index_to_coord(index:int,shape:tuple)-> tuple:
    """Convert a flat index to multi-dimensional coordinates based on the given shape.

    Args:
        index (int): The flat index to convert.
        shape (tuple): The shape of the multi-dimensional array.

    Returns:
        tuple: The multi-dimensional coordinates corresponding to the flat index.
    """
    coord = []
    
    for s in reversed(shape):
        coord.append(index%s)
        index //= s
    coord.reverse()
    return tuple(coord)

def coord_to_index(coord:tuple, shape:tuple)-> int:
    """Convert multi-dimensional coordinates to a flat index based on the given shape.

    Args:
        coord (tuple): The multi-dimensional coordinates to convert.
        shape (tuple): The shape of the multi-dimensional array.

    Returns:
        int: The flat index corresponding to the multi-dimensional coordinates.
    """
    index = 0
    
    for i in range(len(shape)):
        index = index*shape[i] + coord[i]
    return int(index)

def prod(shape):
    p = 1
    for x in shape:
        p *= x
    return p
class Array:
    def __init__(self,data):
        self.data = data
        
        self.shape = self._get_shape(data)
        
        self.ndim = len(self.shape)
        
        self.size = self._get_size(self.shape)
        
        self._LU = None
        
        self._det = None
        
        self._swap_count = 0


    def _get_shape(self, data)-> tuple:
        """Get shape of data using recursive function

        Args:
            data (list, nested listed or scalar): 
                A nested list representing an n-dimensional array. Scalar is the input of the deepest level
        Returns:
            shape (tuple): Shape of the nested list
        """
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            else:
                return (len(data),) + self._get_shape(data[0])
        else:
            return ()
        
    def _get_size(self, shape: tuple)-> int:
        """Compute the total number of elements in an Array from its shape.

        Args:
            shape(tuple): Array's shape

        Returns:
            size(int): Total number of element in an Array
        """
        return prod(shape)

    def get_col(self, index: int)-> Array:
        """Return a column of the 2D Array as a new Array.

        Args:
            index (int): The index of the column to extract

        Raises:
            ValueError: If the Array is not 2-dimensional.
            IndexError: If the column index is out of bounds
        """
        if self.ndim != 2:
            raise ValueError("get_col method only supports 2D arrays")
        if index < 0 or index >= self.shape[1]:
            raise IndexError("Column index out of range")
        
        col_data = [[self.data[i][index]] for i in range(self.shape[0])]
        return Array(col_data)
    
    def set_col(self, index:int, col:Array):
        """Replace a column of the 2D Array with the values from another Array.

        Args:
            index (int): The column index to replace.
            col (Array): A column vector (n,1) whose values will overwrite the
                        specified column of the current Array. 

        Raises:
            ValueError: If the Array is not 2-dimensional.
            IndexError: If the column index is out of bounds
            ValueError: If shape of new column is not matched with the Array shape
        """
        if self.ndim != 2:
            raise ValueError("set_col method only supports 2D arrays")
        if index < 0 or index >= self.shape[1]:
            raise IndexError("Column index out of range")
        if col.shape[0] != self.shape[0]:
            raise ValueError("Column array has incompatible shape")
        
        for i in range(self.shape[0]):
            self.data[i][index] = col.data[i][0]
            
    def get_row(self, index: int):
        if self.ndim != 2:
            raise ValueError("get_row method only supports 2D arrays")
        if index < 0 or index >= self.shape[0]:
            raise IndexError("Row index out of range")
        
        return Array(self.data[index])

    def flatten (self):
        """Flatten an n-dimensional Array into a 1D Python list.

        Returns:
            Flattened Array (list): List A 1D Python list containing all scalar elements of the array,
            in row-major order.
        """
        flattened_array = []
        def _flat(arr):
            for x in arr:
                if not isinstance(x,list):
                    flattened_array.append(x)
                else:
                    _flat(x)
        _flat(self.data)
        return flattened_array


    def reshape(self, new_shape:tuple)-> Array:
        """  Return a new Array with the same data but a different shape.

        Args:
            new_shape (tuple): The desired shape of the output array

        Raises:
            ValueError: If the total number of elements implied by `new_shape` does not
                        match the size of the original array.

        Returns:
            new Array (Array): A new Array instance whose data is the same as the original,
                                but arranged according to `new_shape`.
        """
        # check if the new shape is compatible with the current size
        new_size = self._get_size(new_shape)
        if new_size != self.size:
            raise ValueError(f"Cannot reshape array of size {self.size} into shape {new_shape}.")
        
        # flatten the data
        flat_data = self.flatten()
        
        new_data, _ = build_nested_list(flat_data, new_shape)
        return Array(new_data)
    
    # TODO: understand this function
    def __str__(self):
        """ Return a formatted string representation of the Array.
        """
        def format_array(arr, shape, level=0):
            if len(shape) == 1:
                return '[' + ' '.join(map(str, arr)) + ']' 
            else:
                step = int(len(arr) / shape[0])
                rows = []
                for i in range(shape[0]):
                    part = arr[i*step:(i+1)*step]
                    rows.append(format_array(part, shape[1:], level+1))
                newlines = '\n' * (len(shape) - 1)
                indent = ' ' * (level + 1)
                newline = newlines + indent
                return '[' + newline.join(rows) + ']'
        
        return format_array(self.flatten(), self.shape)
    
    def copy(self):
        """Return a deep copy of an 2-D Array
        """
        new_data = [row[:] for row in self.data]
        return Array(new_data)
    
    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes = None)-> Array:
        """Permute the axes of the Array and return a new transposed Array.
            The operation constructs a new Array by:
                1. Flattening the original data,
                2. Mapping each element's coordinate under the new axis order,
                3. Reshaping the permuted flat list into the new shape.
        Args:
            axes (tuple, optional): Desired ordering of axes. Defaults to None.

        Raises:
            ValueError: If the length of axes does not match the Array.ndim
            ValueError: If the axes does not contain interger from 0 to Array.ndim -1

        Returns:
            Array: A new Array with its axes permuted according to `axes`.
        """
        old_shape = self.shape
        
        if axes is None:
            axes = tuple(x for x in range(self.ndim - 1, -1, -1))

        if len(axes) != self.ndim:
            raise ValueError("Axes don't match array")
        if sorted(axes) != list(range(self.ndim)):
            raise ValueError("Invalid axes for transpose")
        
        new_shape = tuple(old_shape[axis] for axis in axes)
        
        flat = self.flatten()
        new_flat = [0] * len(flat)
        
        for old_index in range(len(flat)):
            old_coord = index_to_coord(old_index, old_shape)
            new_coord = tuple(old_coord[axis] for axis in axes)
            new_index = coord_to_index(new_coord, new_shape)
            new_flat[new_index] = flat[old_index]

        new_data, _ = build_nested_list(new_flat, new_shape)
        return Array(new_data)
    
    ################################################Elementwise Operations#####################################################
    def __add__(self:Array, other:Array)-> Array:
        """ Element-wise addition of two Arrays

        Args:
            self (Array): First Array
            other (Array): Second Array

        Raises:
            ValueError: If the two Arrays do not have the same size.

        Returns:
            Array: A new Array whose elements are the sum of the corresponding elements of self and other.
        """
        self_flat = self.flatten()
        other_flat = other.flatten()
        
        if len(self_flat) != len(other_flat):
            raise ValueError("Arrays must have the same size for addition")
        
        result_flat = [a + b for a, b in zip(self_flat, other_flat)]
        result_data, _ = build_nested_list(result_flat, self.shape)
        # Total complexity =: O(size)
        return Array(result_data)
    
    def __mul__(self:Array, other:float)-> Array:
        """ Element-wise multiplication of an Array by a scalar.

        Args:
            self (Array): The Array to be scaled.
            other (float): The scalar multiplier.

        Returns:
            Array: A new Array whose elements are the elements of self multiplied by other.
        """
        self_flat = self.flatten() # O(size)
        
        result_flat = [a * other for a in self_flat] # O(size)
        result_data, _ = build_nested_list(result_flat, self.shape) # O(size)
        return Array(result_data)
    
    def __sub__(self:Array, other:Array)-> Array:
        """ Element-wise subtraction of two Arrays.

        Args:
            self (Array): The minuend Array.
            other (Array): The subtrahend Array.

        Returns:
            Array: A new Array whose elements are the difference between the corresponding elements of self and other.
        """
        return self + (other * -1)
    
    def __truediv__(self:Array, other:float)-> Array:
        """Element-wise division of an Array by a scalar.

        Args:
            self (Array): The Array to be divided.
            other (float): The scalar divisor.

        Returns:
            Array: A new Array whose elements are the elements of self divided by other.
        """
        self_flat = self.flatten()
        
        result_flat = [a / other for a in self_flat]
        result_data, _ = build_nested_list(result_flat, self.shape)
        # Total complexity: O(sizes)
        return Array(result_data)
    
    def __pow__(self:Array, other:float)-> Array:
        """Element-wise exponentiation of an Array by a scalar.

        Args:
            self (Array): The base Array.
            other (float): The exponent.

        Returns:
            Array: A new Array whose elements are the elements of self raised to the power of other.
        """
        self_flat = self.flatten()
        
        result_flat = [a ** other for a in self_flat]
        result_data, _ = build_nested_list(result_flat,self.shape)
        # Total complexity: O(sizes)
        return Array(result_data)
    
    def __matmul__(self:Array, other:Array)-> Array:
        """ Matrix multiplication of two Arrays. 2D Arrays or 1D Arrays are supported.

        Args:
            self (Array): First Array
            other (Array): Second Array

        Raises:
            ValueError: If the inner dimensions do not match for matrix multiplication.

        Returns:
            Array: A new Array resulting from the matrix multiplication of self and other.
        """
        A = self.data
        B = other.data
        
        if self.ndim == 1:
            A = [A]
        if other.ndim == 1:
            B = [[b] for b in B]
        
        cols_A = len(A[0])
        cols_B = len(B[0])
        rows_A = len(A)
        rows_B = len(B)
        
        if cols_A != rows_B:
            raise ValueError("Inner dimensions must match for matrix multiplication")
        
        result_data = []
        for i in range(rows_A):
            row = []
            for j in range(cols_B):
                sum_product = 0
                for k in range(rows_B):
                    sum_product += A[i][k] * B[k][j]
                row.append(sum_product)
            result_data.append(row)
        
        if self.ndim == 1:
            return Array(result_data[0])
        
        if other.ndim ==1:
            return Array([x[0] for x in result_data])
        # Total complexity: O(mxnxk)~O(n^3)
        return Array(result_data)
    
    def exp(self,cols = None, rows = None)-> Array:
        """Compute the element-wise exponential of the Array.

        Returns:
            Array: A new Array whose elements are the exponential of the corresponding elements of self.
        """
        if cols is None and rows is None:
            data = self
        elif cols is not None and rows is None:
            data = self.get_col(cols)
        elif cols is None and rows is not None:
            data = self.get_row(rows)
        
        self_flat = data.flatten()
        
        result_flat = [math.exp(a) for a in self_flat]
        result_data, _ = build_nested_list(result_flat,data.shape)
        # Total complexity: O(sizes)
        return Array(result_data)
    
    def log(self, cols = None, rows = None)-> Array:
        """Compute the element-wise natural logarithm of the Array.

        Returns:
            _type_: A new Array whose elements are the natural logarithm of the corresponding elements of self.
        """
        if cols is None and rows is None:
            data = self
        elif cols is not None and rows is None:
            data = self.get_col(cols)
        elif cols is None and rows is not None:
            data = self.get_row(rows)
        
        self_flat = data.flatten()
        
        result_flat = [math.log(a) for a in self_flat]
        result_data, _ = build_nested_list(result_flat,data.shape)
        # Total complexity: O(sizes)
        return Array(result_data)
    
    def abs(self, cols = None, rows = None)-> Array:
        """Compute the element-wise absolute value of the Array.

        Returns:
            Array: A new Array whose elements are the absolute values of the corresponding elements of self.
        """
        if cols is None and rows is None:
            data = self
        elif cols is not None and rows is None:
            data = self.get_col(cols)
        elif cols is None and rows is not None:
            data = self.get_row(rows)
        
        self_flat = data.flatten()
        
        result_flat = [abs(a) for a in self_flat]
        result_data, _ = build_nested_list(result_flat,data.shape)
        # Total complexity: O(sizes)
        return Array(result_data)
    
    def sqrt(self, cols = None, rows = None)-> Array:
        """Compute the element-wise square root of the Array.

        Returns:
            Array: A new Array whose elements are the square roots of the corresponding elements of self.
        """
        if cols is None and rows is None:
            data = self
        elif cols is not None and rows is None:
            data = self.get_col(cols)
        elif cols is None and rows is not None:
            data = self.get_row(rows)
        
        self_flat = data.flatten()
        result_flat = [math.sqrt(a) for a in self_flat]
        result_data, _ = build_nested_list(result_flat,data.shape)
        # Total complexity: O(sizes)
        return Array(result_data)
    
    def sum(self, cols = None, rows = None)-> float:
        """Compute the sum of all elements in the Array.

        Returns:
            float: The sum of all elements.
        """
        if cols is None and rows is None:
            self_flat = self.flatten()
        elif cols is not None and rows is None:
            self_flat = self.get_col(cols).flatten()
        elif cols is None and rows is not None:
            self_flat = self.get_row(rows).flatten()
        
        sum = 0
        for i in range(len(self_flat)):
            sum += self_flat[i]
        # Total complexity: O(sizes)
        return float(sum)
    
    def mean(self, cols = None, rows = None)-> float:
        """Compute the mean (average) of all elements in the Array.

        Returns:
            float: The mean of all elements.
        """
        if cols is None and rows is None:
            self_flat = self.flatten()
        elif cols is not None and rows is None:
            self_flat = self.get_col(cols).flatten()
        elif cols is None and rows is not None:
            self_flat = self.get_row(rows).flatten()
        # Total complexity: O(sizes)
        return float(sum(self_flat) / len(self_flat))
    
    def max(self, cols = None, rows = None)-> float:
        """Compute the maximum value among all elements in the Array.

        Returns:
            float: The maximum value.
        """
        if cols is None and rows is None:
            self_flat = self.flatten()
        elif cols is not None and rows is None:
            self_flat = self.get_col(cols).flatten()
        elif cols is None and rows is not None:
            self_flat = self.get_row(rows).flatten()
            
        max = self_flat[0] 
        for i in range(1, len(self_flat)):
            if self_flat[i] > max:
                max = self_flat[i]
        # Total complexity: O(sizes)
        return float(max)
    
    def min(self, cols = None, rows = None)-> float:
        """Compute the minimum value among all elements in the Array.

        Returns:
            float: The minimum value.
        """
        if cols is None and rows is None:
            self_flat = self.flatten()
        elif cols is not None and rows is None:
            self_flat = self.get_col(cols).flatten()
        elif cols is None and rows is not None:
            self_flat = self.get_row(rows).flatten()
            
        min = self_flat[0] 
        for i in range(1, len(self_flat)):
            if self_flat[i] < min:
                min = self_flat[i]
        # Total complexity: O(sizes)
        return float(min)

    def arg_min(self, cols = None, rows = None)-> tuple:
        """Find the indices of the minimum value in the Array.

        Returns:
            tuple: The indices of the minimum value as a tuple.
        """
        if cols is None and rows is None:
            data = self
        elif cols is not None and rows is None:
            data = self.get_col(cols)
        elif cols is None and rows is not None:
            data = self.get_row(rows)
            
        self_flat = data.flatten()
        a = min(self_flat)
        for i in range(len(self_flat)):
            if self_flat[i] == a:
                # Total complexity: O(sizes)
                return index_to_coord(i,data.shape)
            
    def arg_max(self, cols = None, rows = None)-> tuple:
        """Find the indices of the maximum value in the Array.

        Returns:
            tuple: The indices of the maximum value as a tuple.
        """
        if cols is None and rows is None:
            data = self
        elif cols is not None and rows is None:
            data = self.get_col(cols)
        elif cols is None and rows is not None:
            data = self.get_row(rows)
            
        self_flat = data.flatten()
        a = max(self_flat)
        for i in range(len(self_flat)):
            if self_flat[i] == a:
                # Total complexity: O(sizes)
                return  index_to_coord(i,data.shape)
            
    def LU_Decomposition(self):
        """Perform LU Decomposition of a square Array.
            Idea: PA = LU
            where P is permutation matrix
            L is lower triangular matrix with unit diagonal elements
            and U is upper triangular matrix.

        Raises:
            ValueError: If the Array is not square.
            ValueError: If a zero pivot is encountered (no pivoting implemented).

        Returns:
            L (Array): Lower triangular matrix with unit diagonal elements.
            U (Array): Upper triangular matrix.
        """
        if self._LU is not None:
            return self._LU
        
        n = self.shape[0]
        
        U = self.copy()
        swap_count = 0
        L = eye(n)
        P = eye(n)
        
        for p in range(n):
            pivot_row = p 
            max_val = abs(U.data[p][p])
            for i in range (p+1, n):
                if abs(U.data[i][p]) > max_val:
                    max_val = abs(U.data[i][p])
                    pivot_row = i
                    
            if U.data[pivot_row][p] == 0:
                raise ValueError("Zero pivot encountered — pivoting required")

            if pivot_row != p:
                U.data[pivot_row], U.data[p] = U.data[p],U.data[pivot_row]
                P.data[pivot_row], P.data[p] = P.data[p],P.data[pivot_row]
                swap_count ^= 1
                
                if p>0:
                    L.data[p][:p], L.data[pivot_row][:p] =  L.data[pivot_row][:p], L.data[p][:p]
            
            pivot = U.data[p][p]         
            
            for i in range(p+1,n):
                w = U.data[i][p]/pivot
                L.data[i][p] = w
                
                for j in range(p,n):
                    U.data[i][j] -= w*U.data[p][j]
                    
        self._swap_count = swap_count
        self._LU = (P, L, U)
        return P, L, U
    
    def determinant(self)-> float:
        """Compute the determinant of a square Array using its LU Decomposition.

        Returns:
            float: The determinant of the Array.
        """
        if self._det is not None:
            return self._det
        
        try:
            P, L, U = self.LU_Decomposition()
        except ValueError:
            # LU failed → matrix is singular
            self._det = 0.0
            return 0.0
        
        det = 1
        for i in range(self.shape[0]):
            det*= U.data[i][i]
        
        self._det = (-1) **self._swap_count * det
        return (-1) **self._swap_count * det
    
    def det_Bareiss(self):
        if self.ndim != 2:
            raise ValueError("Matrix should be 2-D")
        if self.shape[0] != self.shape[1]:
            raise ValueError("Matrix must be square")
        
        n = self.shape[0]
        A = self.copy()
        swap_count = 0
        
        for p in range(n):
            
            if A.data[p][p] == 0:
                found = False
                for r in range(p+1, n):
                    if A.data[r][p] != 0:
                        A.data[p], A.data[r] = A.data[r], A.data[p]
                        swap_count ^= 1
                        found = True
                        break
                if not found:
                    return 0 
            
            pivot = 1 if p == 0 else A.data[p-1][p-1]
            if pivot == 0:
                return 0
            
            for i in range(p+1,n):
                for j in range(p+1,n):
                    A.data[i][j] = (A.data[p][p]*A.data[i][j] - A.data[i][p]*A.data[p][j])/pivot
                A.data[i][p] = 0
            
        det = A.data[n-1][n-1]
        
        if swap_count == 1:
            det = -det
        
        return det           
        
