import numpy as np

def get_long_training_sequence(step=1):
    """
    Generiše Long Training Sequence u vremenskom domenu.
    """

    # Definicije tona iz MATLAB-a
    Positive = np.array([
        0, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,
       -1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1
    ], dtype=complex)

    Negative = np.array([
        0, 1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1, 1,-1, 1,-1,
       -1, 1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1, 1, 1, 1
    ], dtype=complex)

    # MATLAB: AllTones = [Negative Positive];
    AllTones = np.concatenate((Negative, Positive))

    N = 64
    m = np.arange(-32, 32)  # MATLAB m = -32:31

    # MATLAB: LongTrainingSymbol = zeros(1, 64/Step)
    length = int(64 / step)
    LongTrainingSymbol = np.zeros(length, dtype=complex)

    # MATLAB IDFT petlja
    for n in range(length):
        t = n * step
        E = np.exp(1j * 2 * np.pi * t * m / N)
        LongTrainingSymbol[n] = np.dot(AllTones, E)  # BEZ 1/N !!

    # MATLAB CP dodavanje
    if step == 1:
        # MATLAB: LongTrainingSymbol(1,33:64)
        cp = LongTrainingSymbol[32:64]
        return np.concatenate((cp, LongTrainingSymbol))

    else:
        # MATLAB: LongTrainingSymbol(1,65:128)
        cp = LongTrainingSymbol[64:128]
        return np.concatenate((cp, LongTrainingSymbol))
