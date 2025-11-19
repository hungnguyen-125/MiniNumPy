from __future__ import annotations
import math
class Array:
    def __init__(self,data):
        self.data = data
        
        self.shape = self._get_shape(data)
        
        self.ndim = len(self.shape)
        
        self.size = self._get_size(self.shape)

        self._LU = None
        
        self._det = None
        
    def _get_shape(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                return (0,)
            else:
                return (len(data),) + self._get_shape(data[0])
        else:
            return ()
        
    def _get_size(self, shape):
        size = 1
        for dim in shape:
            size *= dim
        return size
    
    def flatten (self):
        flattened_array = []
        def _flat(arr):
            for x in arr:
                if not isinstance(x,list):
                    flattened_array.append(x)
                else:
                    _flat(x)
        _flat(self.data)
        return flattened_array
    
    # def build(self, shape:tuple):
    #     flat = self.flatten()
    #     relsult = []
        
    #     size = 1
    #     for x in shape:
    #         size*= x
            
    #     if size != len(flat):
    #         raise ValueError("Matrix dimension violation")
    #     for i in range(len(shape),0,-1):
    #         for j in range(len(shape)/shape[i]):
    #             relsult.append(flat[shape[i]*j:shape[i]*(j+1)-1])
    #         flat = relsult
            
    #     return flat
            
    #TODO: seprate build_nested_list from reshape method
    def reshape(self, new_shape):
        # check if the new shape is compatible with the current size
        new_size = self._get_size(new_shape)
        if new_size != self.size:
            raise ValueError("Cannot reshape array of size {} into shape {}".format(self.size, new_shape))
        
        # flatten the data
        flat_data = self.flatten()
        
        new_data, _ = _build_nested_list(flat_data, new_shape)
        return Array(new_data)
    
    # TODO: understand this function
    def __str__(self):
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
        # deep copy the underlying data
        new_data = [row[:] for row in self.data]
        return Array(new_data)

    #TODO: add transpose method for nD arrays
    def transpose(self):
        """
        Transpose an n-dimensional array by reversing the order of axes.
        Returns a new Array with axes transposed.
        """
        if not isinstance(self.data, list):
            return self

        # Helper function to get item at specific indices
        def get_item(data, indices):
            current = data
            for idx in indices:
                current = current[idx]
            return current

        # Helper function to create transposed data structure
        def build_transposed(shape):
            if not shape:
                return None
            return [build_transposed(shape[1:]) for _ in range(shape[0])]

        # Get all possible indices for the transposed array
        def get_indices(shape):
            if not shape:
                return [[]]
            result = []
            for i in range(shape[0]):
                for sub_indices in get_indices(shape[1:]):
                    result.append([i] + sub_indices)
            return result

        # Create new shape by reversing axes
        new_shape = self.shape[::-1]
        
        # Create empty nested structure for transposed data
        transposed_data = build_transposed(new_shape)
        
        # Fill the transposed data
        for indices in get_indices(self.shape):
            # Reverse the indices for the transposed array
            transposed_indices = indices[::-1]
            
            # Get the value from original array
            value = get_item(self.data, indices)
            
            # Set the value in the transposed array
            current = transposed_data
            for idx in transposed_indices[:-1]:
                current = current[idx]
            current[transposed_indices[-1]] = value

        return Array(transposed_data)
    
    ################################################Elementwise Operations#####################################################
    def __add__(self:Array, other:Array)-> Array:
        self_flat = self.flatten()
        other_flat = other.flatten()
        
        if len(self_flat) != len(other_flat):
            raise ValueError("Arrays must have the same size for addition")
        
        result_flat = [a + b for a, b in zip(self_flat, other_flat)]
        result_data, _ = _build_nested_list(result_flat, self.shape)
        return Array(result_data)
    
    def __mul__(self:Array, other:float)-> Array:
        self_flat = self.flatten()
        
        result_flat = [a * other for a in self_flat]
        result_data, _ = _build_nested_list(result_flat, self.shape)
        return Array(result_data)
    
    def __sub__(self:Array, other:Array)-> Array:
        return self + (other * -1)
    
    def __matmul__(self:Array, other:Array)-> Array:
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Both arrays must be 2-dimensional for matrix multiplication")
        if self.shape[1] != other.shape[0]:
            raise ValueError("Inner dimensions must match for matrix multiplication")
        
        result_data = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                sum_product = 0
                for k in range(self.shape[1]):
                    sum_product += self.data[i][k] * other.data[k][j]
                row.append(sum_product)
            result_data.append(row)
        
        return Array(result_data)
    
    def __truediv__(self:Array, other:float)-> Array:
        self_flat = self.flatten()
        
        result_flat = [a / other for a in self_flat]
        result_data, _ = _build_nested_list(result_flat, self.shape)
        return Array(result_data)
    
    def __pow__(self:Array, other:float):
        self_flat = self.flatten()
        
        result_flat = [a ** other for a in self_flat]
        result_data, _ = _build_nested_list(result_flat,self.shape)
        return Array(result_data)
    
    def exp(self):
        self_flat = self.flatten()
        
        result_flat = [math.exp(a) for a in self_flat]
        result_data, _ = _build_nested_list(result_flat,self.shape)
        return Array(result_data)
    
    def log(self):
        self_flat = self.flatten()
        
        result_flat = [math.log(a) for a in self_flat]
        result_data, _ = _build_nested_list(result_flat,self.shape)
        return Array(result_data)
    
    def abs(self):
        self_flat = self.flatten()
        
        result_flat = [math.abs(a) for a in self_flat]
        result_data, _ = _build_nested_list(result_flat,self.shape)
        return Array(result_data)
    
    def sqrt(self):
        self_flat = self.flatten()
        
        result_flat = [math.sqrt(a) for a in self_flat]
        result_data, _ = _build_nested_list(result_flat,self.shape)
        return Array(result_data)
    
    def sum(self):
        self_flat = self.flatten()
        sum = 0
        for i in range(len(self_flat)):
            sum += self_flat[i]
        return sum
    
    def mean(self):
        self_flat = self.flatten()
        return sum(self_flat) / len(self_flat)
    
    def max(self):
        self_flat = self.flatten()
        max = self_flat[0] 
        for i in range(1, len(self_flat)):
            if self_flat[i] > max:
                max = self_flat[i]
        return max
    
    def min(self):
        self_flat = self.flatten()
        min = self_flat[0] 
        for i in range(1, len(self_flat)):
            if self_flat[i] < min:
                min = self_flat[i]
        return min

    def arg_min(self):
        self_flat = self.flatten()
        a = min(self_flat)
        for i in range(len(self_flat)):
            if self_flat[i] == a:
                return i
    
    def arg_max(self):
        self_flat = self.flatten()
        a = max(self_flat)
        for i in range(len(self_flat)):
            if self_flat[i] == a:
                return i
    
    #TODO: calculate P - permutation matrix
    def LU_Decomposition(self):
        if self._LU is not None:
            return self._LU
        
        n = self.shape[0]
        if n != self.shape[1]:
            raise ValueError("LU decomposition requires a square matrix")

        L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        U = self.copy()

        for p in range (n-1):
            pivot = U.data[p][p]
            if pivot == 0:
                raise ValueError("Zero pivot encountered — pivoting required")

            for i in range(p+1, n):
                w = U.data[i][p]/pivot
                L[i][p] = w
                
                for j in range (p, n):
                    U.data[i][j] -= w*U.data[p][j]
        self._LU = (L, U)
        return Array(L), U   
    
    def determinant(self):
        if self._det is not None:
            return self._det
        
        L, U = self.LU_Decomposition()
        det = 1
        for i in range(self.shape[0]):
            det*= U.data[i][i]
        
        self._det = det
        return det 


# build the new nested list structure
def _build_nested_list(flat_data, shape):
    if len(shape) == 0:
        return flat_data[0], flat_data[1:]
    else:
        dim = shape[0]
        sub_shape = shape[1:]
        nested_list = []
        for _ in range(dim):
            sub_list, flat_data = _build_nested_list(flat_data, sub_shape)
            nested_list.append(sub_list)
        return nested_list, flat_data
    
def array(data):
    return Array(data)

def _fill(shape:tuple, fill_value):
    if len(shape) == 0:
        return fill_value
    else:
        return [_fill(shape[1:], fill_value) for _ in range(shape[0])]

def zeros(shape:tuple):
    data = _fill(shape, 0)
    return Array(data)

def ones(shape:tuple):
    data = _fill(shape, 1)
    return Array(data)

def identity(n:int):
    data = _fill((n, n), 0)
    for i in range(n):
        data[i][i] = 1
    return Array(data)

def eye(n: int, m = None, k: int = 0):
    if m is None: 
        m = n
        
    data = _fill((n, m), 0)
    for i in range(min(n)):
        j = i + k
        if j < m:
            data[i][j] = 1
    return Array(data)

def arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    data = []
    value = start
    while (step > 0 and value < stop) or (step < 0 and value > stop):
        data.append(value)
        value += step
    return Array(data)

def linspace(start, stop, num=50):
    if num <= 0:
        return Array([])
    if num == 1:
        return Array([start])
    step = (stop - start) / (num - 1)
    data = [start + i * step for i in range(num)]
    return Array(data)

