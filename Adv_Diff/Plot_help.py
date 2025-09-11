import numpy as np

def sci_notation(x):
    """Return a string in the format 'a × 10^b' for a given number x."""
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exponent
    return rf"{mantissa:.2f} $\times 10^{{{exponent}}}$"
