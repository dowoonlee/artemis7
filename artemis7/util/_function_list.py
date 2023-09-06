from inspect import getmembers, isfunction, isclass, getfullargspec
from tabulate import tabulate
from ._colorprint import Colorprint as cp

__all__ = ["list"]

def list(class_):
    """
    class_ : target class to get list of functions of it
    """
    member_candidates = getmembers(class_, isfunction)
    ml = []
    print(cp.BOLD + "Functions" + cp.END)
    for mc in member_candidates:
        if class_.__name__ == mc[1].__module__:
            ml.append([mc[0], ", ".join(getfullargspec(mc[1]).args)])
    if not ml:
        print(cp.RED + "None" + cp.END)
    else:
        print(tabulate(ml))
    
    member_candidates = getmembers(class_, isclass)
    ml = []
    print(cp.BOLD + "Classes" + cp.END)
    for mc in member_candidates:
        if class_.__name__ == mc[1].__module__:
            ml.append([mc[0], ", ".join(getfullargspec(mc[1]).args)])
    if not ml:
        print(cp.RED + "None" + cp.END)
    else:
        print(tabulate(ml))
    return