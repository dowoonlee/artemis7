import numpy as np
import time

class progressbar():
    def __init__(self, maxV):
        self.maxV = maxV
        self.current_lim = 1
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
            return "ETA %d:%02d"%(h, m)
        return "ETA %02d:%02d"%(m, s)

    def _pout(self, n, percent, eta_disp=True):
        bk = 4-len(str(int(percent)))
        n//=2
        bar = "| "  + "█"*n + "#"*(50-n)
        percentage = " "*bk +"%d%%"%percent+ "|"
        if eta_disp:
            eta = self._ETA(percent)
            return  bar + percentage + eta
        return bar + percentage

    def start(self):
        print(self._pout(0, 0), end="")
        self.t0 = time.time()

    def update(self, c):
        current_progress = c/self.maxV*100
        if np.ceil(current_progress)>=self.current_lim:
            print("\r"+self._pout(self.current_lim, current_progress), end="")
            self.current_lim += 1

    def finish(self):
        print("\r"+self._pout(100, 100, eta_disp=False), end="")
        print(" Finish    ")
        return
