import math

class Array:
    def __init__(self,data):
        self.data = data
        
        self.shape = self._get_shape(data)
        
        self.ndim = len(self.shape)
        
        self.size = self._get_size(self.shape)
    
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
    
    #TODO: seprate build_nested_list from reshape method
    def reshape(self, new_shape):
        # check if the new shape is compatible with the current size
        new_size = self._get_size(new_shape)
        if new_size != self.size:
            raise ValueError("Cannot reshape array of size {} into shape {}".format(self.size, new_shape))
        
        # flatten the data
        flat_data = self.flatten()
        
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
    
    def flatten(self):
        # flatten the nested Python list in self.data and return a flat list of values
        def _flatten(data):
            if not isinstance(data, list):
                return [data]
            res = []
            for item in data:
                res.extend(_flatten(item))
            return res

        return _flatten(self.data)
    
def array(data):
    return Array(data)

def _build(shape, fill_value):
    if len(shape) == 0:
        return fill_value
    else:
        return [_build(shape[1:], fill_value) for _ in range(shape[0])]
    
def zeros(shape):
    data = _build(shape, 0)
    return Array(data)

def ones(shape):
    data = _build(shape, 1)
    return Array(data)

def eye(n):
    data = _build((n, n), 0)
    for i in range(n):
        data[i][i] = 1
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
