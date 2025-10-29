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
    
    # TODO: add str method for pretty printing
    def __str__(self):
        return str(self.data)
    
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
