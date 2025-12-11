import numpy as np

def GetMultipathFilter(SampleRate, DelaySpread, N):
    Ts = 1 / SampleRate  # Sampling Period in seconds
    Trms = DelaySpread   # Delay spread in seconds

    n = np.arange(N)

    ExpVariance = np.exp(-n * Ts / Trms)
    FIR_Taps = np.zeros(N, dtype=complex)

    for i in range(N):
        FIR_Taps[i] = np.sqrt(ExpVariance[i]) * np.random.randn() + 1j * np.sqrt(ExpVariance[i]) * np.random.randn()

    return FIR_Taps