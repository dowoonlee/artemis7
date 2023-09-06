import numpy as np
import time

__all__ = ["ProgressBar"]

class ProgressBar():
    def __init__(self, maxV, jobname=""):
        self.maxV = maxV
        self._current_percent = 1
        self.t0 = time.time()
        self._banner_size = 10
        self.jobname = "."*(max([self._banner_size-len(jobname), 1]))+jobname
        
    def start(self):
        print(self._pout(0), end="")
        self.t0 = time.time()
    
    def _ETA(self, progress):
        t_now = time.time()
        t_passed = t_now - self.t0
        eta = t_passed * ((100-progress)/(progress+1e-6))
        h = eta//3600
        eta -= h*3600
        m = eta//60
        s = eta%60
        if h>0:
            return "ETA %dH:%02dm"%(h, m)
        return "ETA %02dm:%02ds"%(m, s)

    def _pout(self, percent, eta_disp=True):
        bk = 4-len(str(int(percent)))
        bar = "| "  + "█"*int(percent//2) + " "*int(50-percent//2)
        percentage = " "*bk +"%d%%"%percent+ "|"
        if eta_disp:
            eta = self._ETA(percent)
            return  bar + percentage + eta + self.jobname
        return bar + percentage + self.jobname

    def update(self, c):
        percent = c/self.maxV*100
        if abs(self.maxV-c) <= 1:
            self.__del__()
        if np.ceil(percent)>=self._current_percent:
            print("\r"+self._pout(percent), end="")
            self._current_percent = percent
    
    def __del__(self):
        print("\r"+self._pout(100, eta_disp=True), end="")

