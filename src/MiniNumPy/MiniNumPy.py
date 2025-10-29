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
    