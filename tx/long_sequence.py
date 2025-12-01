import numpy as np

def get_long_training_sequence(step=1):
    """
    Generiše Long Training Sequence (802.11a) u vremenskom domenu.
    Long sekvenca se koristi za channel estimation i fine CFO korekciju.
    """
    
    Positive = np.array([0,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,1,
                         -1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,1])

    Negative = np.array([0,1,1,-1,1,-1,1,-1,1,1,-1,1,1,-1,1,-1,
                         -1,1,1,-1,1,-1,1,-1,1,-1,1,-1,1,1,1,1])

    AllTones = np.concatenate((Negative, Positive))

    N = 64
    m = np.arange(-32, 32)

    length = int(64 / step)
    LongTrainingSymbol = np.zeros(length, dtype=complex)

    for n in range(length):
        t = n * step
        E = np.exp(1j * 2 * np.pi * t * m / N)
        LongTrainingSymbol[n] = np.dot(AllTones, E)

    if step == 1:   # 20 MHz
        return np.concatenate((LongTrainingSymbol[32:64], LongTrainingSymbol))
    else:           # 40 MHz (step=0.5 → dvostruko više tačaka)
        return np.concatenate((LongTrainingSymbol[64:128], LongTrainingSymbol))
