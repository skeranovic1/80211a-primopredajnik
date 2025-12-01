import numpy as np

def get_short_training_sequence(step=1):
    """
    Generiše Short Training Sequence (802.11a) u vremenskom domenu
    koristeći IDFT formulu.
    Short sekvenca se koristi za packet detection, AGC, i coarse CFO.
    """
    Positive = np.array([0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,
                         1+1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0])

    Negative = np.array([0,0,0,0,0,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,
                         1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0])

    Total = np.sqrt(13/6) * np.concatenate((Negative, Positive))
    
    N = 64
    m = np.arange(-32, 32)

    length = int(160 / step)
    ShortTrainingSequence = np.zeros(length, dtype=complex)

    for n in range(length):
        t = n * step
        E = np.exp(1j * 2 * np.pi * t * m / N)
        ShortTrainingSequence[n] = np.dot(Total, E)

    return ShortTrainingSequence
