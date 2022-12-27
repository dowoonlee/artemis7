import numpy as np
from scipy.io import FortranFile as FF

class Binary_IO():
    def __init__(self, mode, **kwargs):
        if mode == "read":
            file = kwargs["file"]
            self._f = FF(file, mode="r")
        
        elif mode == "write":
            file = kwargs["file"]
            self._f = FF(file, mode="w")

    def readLine(self, format=np.int32):
        ints = [
            np.int, np.int8, np.int16, np.int32, np.int64,
            np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
            np.int_, np.intp, np.intc]
        floats = [
            np.float_, np.float16, np.flot32, np.float64, np.float128,
            np.single, np.double
        ]

        if format in ints:
            return self._f.read_ints(format)
        elif format in floats:
            return self._f.read_reals(format)
    
    def readAll(self, format=np.int32):
        ints = [
            np.int, np.int8, np.int16, np.int32, np.int64,
            np.uint, np.uint8, np.uint16, np.uint32, np.uint64,
            np.int_, np.intp, np.intc]
        floats = [
            np.float_, np.float16, np.flot32, np.float64, np.float128,
            np.single, np.double
        ]
        if format in ints:
            return