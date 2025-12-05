import numpy as np

def get_short_training_sequence(step=1):
    """
    Generiše Short Training Sequence (STS) za 802.11a standard u vremenskom domenu.

    Short Training Sequence se koristi za:
    - Detekciju paketa (packet detection)
    - Automatsku kontrolu pojačanja (AGC)
    - Grubu korekciju frekvencijskog offseta (coarse CFO)

    Parametri
    step : int, opcionalno
        Korak uzorkovanja u vremenu. Default je 1.

    Povratna vrijednost
    ShortTrainingSequence : numpy.ndarray
        Niz kompleksnih uzoraka u vremenskom domenu koji predstavlja STS.

    Napomene
    - Sekvenca se generiše korištenjem IDFT formule preko unaprijed definisanih frekvencijskih komponenti.
    - 'Positive' i 'Negative' predstavljaju pozitivne i negativne frekvencijske tonove STS.
    - Rezultat se normalizuje faktorom sqrt(13/6).
    - Funkcija vraća kompleksnu vremensku sekvencu dužine proporcionalne parametru 'step'.
    """
    if not isinstance(step, (int, float)):
        raise TypeError(f"Step mora biti numerički tip, dobijeno: {type(step)}")
    if step <= 0:
        raise ValueError(f"Step mora biti pozitivan, dobijeno: {step}")
    
    # Definisanje pozitivnih frekvencijskih komponenti STS
    Positive = np.array([
        0,0,0,0,   -1-1j,0,0,0,   -1-1j,0,0,0,   1+1j,0,0,0,
        1+1j,0,0,0, 1+1j,0,0,0,  1+1j,0,0,0,  0,0,0,0
    ], dtype=complex)

    #Definisanje negativnih frekvencijskih komponenti STS
    Negative = np.array([
        0,0,0,0,   0,0,0,0,   1+1j,0,0,0,   -1-1j,0,0,0,
        1+1j,0,0,0,  -1-1j,0,0,0,  -1-1j,0,0,0,  1+1j,0,0,0
    ], dtype=complex)

    #Kombinovanje negativnih i pozitivnih tonova i normalizacija
    Total = np.sqrt(13/6) * np.concatenate((Negative, Positive))

    #Definisanje parametara IDFT-a
    N = 64
    m = np.arange(-32, 32)

    #Kreiranje praznog niza za vremensku sekvencu
    length = int(160 / step)
    ShortTrainingSequence = np.zeros(length, dtype=complex)

    #Generisanje Short Training Sequence korištenjem IDFT
    for n in range(length):
        t = n * step
        E = np.exp(1j*2*np.pi*t*m/N)
        ShortTrainingSequence[n] = np.dot(Total, E)

    return ShortTrainingSequence
