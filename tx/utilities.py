import numpy as np
import matplotlib.pyplot as plt

def zero_stuffing(signal, up_factor=2):
    """
    Funkcija za zero-stuffing (upsampling) signala.
    
    Ubacuje nule između originalnih uzoraka signala da bi se postiglo upsampliranje.

    Parametri
    signal : numpy.ndarray
        Ulazni signal koji se upsamplira. Mora biti numpy niz.
    up_factor : int, opcionalno
        Faktor upsampliranja (koliko puta se povećava broj uzoraka). Default je 2.

    Povratna vrijednost
    upsamplirano : numpy.ndarray
        Upsamplirani signal sa ubačenim nulama. Tip podataka je kompleksan (complex).

    Izuzeci
    TypeError
        Ako 'signal' nije numpy niz ili 'up_factor' nije cijeli broj.
    ValueError
        Ako 'up_factor' nije pozitivan cijeli broj.
    
    Napomene
    - Funkcija radi tako što kreira novi niz dužine 'len(signal) * up_factor'.
    - Originalni uzorci se stavljaju na pozicije '0, up_factor, 2*up_factor, ...'.
    - Ostale pozicije su popunjene nulama.
    """
    # Provjera tipa ulaznog signala
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal mora biti numpy niz.")

    # Provjera tipa up_factor
    if not isinstance(up_factor, int):
        raise TypeError("up_factor mora biti cijeli broj.")

    # Provjera vrijednosti up_factor
    if up_factor <= 0:
        raise ValueError("up_factor mora biti pozitivan cijeli broj.")

    # Kreiranje upsampliranog signala sa nulama
    upsamplirano = np.zeros(len(signal) * up_factor, dtype=complex)

    # Smještanje originalnih uzoraka na odgovarajuće pozicije
    upsamplirano[::up_factor] = signal
    
    return upsamplirano

def bit_sequence (NumberOf_OFDM_Symbols, BitsPerSymbol, sd=0):
    """
    Generiše nasumičnu bit sekvencu za OFDM prijenos.

    Parametri
    NumberOf_OFDM_Symbols : int
        Broj OFDM simbola za koje se generišu bitovi.
    BitsPerSymbol : int
        Broj bitova po OFDM simbolu.
    sd : int, opcionalno
        Sjeme (seed) za generator nasumičnih brojeva (default je 0).

    Povratna vrijednost
    Source_Bits : numpy.ndarray
        1D niz nasumično generisanih bitova (0 ili 1).

    Napomene
    - Ukupan broj bitova se računa kao 48 * BitsPerSymbol * NumberOf_OFDM_Symbols.
    - 'np.random.seed(sd)' se koristi za reproduktivnost rezultata.
    - Bitovi se generišu korištenjem uniformne distribucije i zaokružuju na cijele brojeve (0 ili 1).
    """
    #Fiksiranje sjemena
    np.random.seed(sd)  

    #Izračunavanje ukupnog broja bitova
    NumberOfBits=(48*BitsPerSymbol)*NumberOf_OFDM_Symbols

    #Generisanje nasumičnih bita (0 ili 1)
    Source_Bits=np.round(np.random.rand(NumberOfBits)).astype(int)
    
    return Source_Bits

def spektar(x, fs, label):
    """
    Funkcija za crtanje spektra signala.

    Parametri
    x : numpy.ndarray
        Ulazni signal čiji se spektar crta.
    fs : float
        Frekvencija uzorkovanja signala u Hz.
    label : str
        Oznaka koja se koristi u legendi grafa.

    Povratna vrijednost
    None
        Funkcija crta spektar signala koristeći matplotlib i ne vraća vrijednost.

    Napomene
    - Spektar se računa pomoću Fast Fourier Transform (FFT) funkcije.
    - Funkcija koristi 'np.fft.fft' za FFT i 'np.fft.fftshift' da centrirano prikaže frekvencijski spektar.
    - Magnituda se normalizuje sa faktorom N/2.
    - Za prikazivanje koristiti 'plt.show()' nakon funkcije da bi se graf prikazao.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Ulaz x mora biti numpy array")
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError("Input x must be numeric")
    if fs <= 0:
        raise ValueError("Sample rate fs mora biti pozitivan")

    # Dužina signala
    N = len(x)

    # Računanje FFT-a i pomjeranje nule u centar spektra
    X = np.fft.fftshift(np.fft.fft(x))

    # Generisanje vektora frekvencija (centrovan na 0 Hz)
    f = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))

    # Računanje magnitude spektra i normalizacija
    magnitude = np.abs(X) / (N/2)

    # Crtanje spektra
    plt.plot(f, magnitude, label=label)

def plot_konstelaciju(symbols,title):
    """
    Crta konstelacioni dijagram za kompleksne simbole.

    Parametri:
    - symbols : np.ndarray
        Niz kompleksnih simbola (npr. izlaz iz Mapper_OFDM)
    - title : str, optional
        Naslov grafa (default: "Konstelacioni dijagram")
    """
    if not isinstance(symbols, np.ndarray):
        raise TypeError("Input symbols must be a numpy array")
    if not np.iscomplexobj(symbols):
        raise ValueError("Input symbols must be complex numbers")

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(symbols), np.imag(symbols), s=10)

    # Ose kroz nulu radi lakše čitanja
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)

    plt.xlabel("I komponenta")
    plt.ylabel("Q komponenta")
    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect("equal", "box")  # da bude krug -> kvadrat
    plt.tight_layout()    