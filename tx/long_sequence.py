import numpy as np

def get_long_training_sequence(step=1):
    """
    Generiše Long Training Sequence (LTS) u vremenskom domenu za OFDM (802.11a standard).

    Long Training Sequence se koristi za:
    - Preciznu sinhronizaciju paketa
    - Kanalnu estimaciju (Channel Estimation)

    Parametri
    step : int, opcionalno
        Korak uzorkovanja u vremenu. Default je 1.

    Povratna vrijednost
    LongTrainingSymbol : numpy.ndarray
        Kompleksna vremenska sekvenca Long Training Sequence, uključujući Cyclic Prefix (CP).

    Napomene
    - Ova funkcija vraća samo long sekvencu koja se kod okvira 802.11a standarda sastoji od ciklicnog prefiksa 
      i dvije uzastopne LTS sekvence.
    - Sekvenca se generiše korištenjem IDFT formule preko definisanih frekvencijskih tonova.
    - 'Positive' i 'Negative' predstavljaju pozitivne i negativne frekvencijske tonove LTS.
    - Cyclic Prefix (CP) se dodaje kako bi se olakšala sinhronizacija i zaštita od inter-symbol interference (ISI).
    - Funkcija vraća kompleksnu vremensku sekvencu čija dužina zavisi od parametra 'step'.
    """

    #Definisanje pozitivnih frekvencijskih komponenti LTS
    Positive = np.array([
       0, 1,-1,-1,   1, 1,-1, 1,  -1, 1,-1,-1,  -1,-1,-1, 1,
       1,-1,-1, 1,  -1, 1,-1, 1,   1, 1, 1, 0,   0, 0, 0, 0
    ], dtype=complex)

    #Definisanje negativnih frekvencijskih komponenti LTS
    Negative = np.array([
       0, 0, 0, 0,  0, 0, 1, 1,  -1,-1, 1, 1,  -1, 1,-1, 1,
       1, 1, 1, 1,  1,-1,-1, 1,   1,-1, 1,-1,   1, 1, 1, 1
    ], dtype=complex)

    #Kombinovanje negativnih i pozitivnih tonova
    AllTones = np.concatenate((Negative, Positive))

    #Parametri IDFT-a
    N = 64
    m = np.arange(-32, 32)  # MATLAB m = -32:31

    #Kreiranje praznog niza za vremensku sekvencu
    length = int(64/step)
    LongTrainingSymbol = np.zeros(length, dtype=complex)

    #IDFT petlja
    for n in range(length):
        t=n*step
        E=np.exp(1j*2*np.pi*t*m/N)
        LongTrainingSymbol[n] = np.dot(AllTones,E) 

    double_long = np.concatenate((LongTrainingSymbol, LongTrainingSymbol))

    #Dodavanje Cyclic Prefix (CP) prema step-u
    if step == 1:
        cp = double_long[32:64] #uzimanje zadnja 32 uzorka
        return np.concatenate((cp, double_long))

    else:
        cp = double_long[64:128] #uzimanje zadnja 64 uzorka
        return np.concatenate((cp, double_long))
