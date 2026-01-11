import pytest
import numpy as np
from rx.rastavljanje import remove_cp  

def test_remove_cp_basic():
    """
    Testira osnovnu funkcionalnost remove_cp:
    - uklanja CP iz više simbola
    - vraća FFT po simbolima
    """
    NFFT = 8
    NCP = 2
    NSYM = NFFT + NCP  # 10
    num_symbols = 3
    start = 0

    # signal dovoljno dug za sve simbole
    rx = np.arange(NSYM*num_symbols) + 1j*np.arange(NSYM*num_symbols)

    symbols_fd = remove_cp(rx, start, num_symbols, NSYM, NFFT, NCP)

    assert symbols_fd.shape == (num_symbols, NFFT)
    for k in range(num_symbols):
        sym_td = rx[start + k*NSYM + NCP : start + k*NSYM + NCP + NFFT]
        expected_fft = np.fft.fft(sym_td)
        np.testing.assert_allclose(symbols_fd[k], expected_fft)

def test_remove_cp_start_within_cp():
    """
    Provjerava da funkcija radi i kada 'start' indeks počinje unutar CP
    """
    NFFT = 4
    NCP = 2
    NSYM = NFFT + NCP  # 6
    num_symbols = 2
    start = 1
    rx = np.arange(NSYM*num_symbols + start) + 1j*np.arange(NSYM*num_symbols + start)
    
    symbols_fd = remove_cp(rx, start, num_symbols, NSYM, NFFT, NCP)
    assert symbols_fd.shape == (num_symbols, NFFT)

def test_remove_cp_insufficient_samples():
    """
    Provjerava da funkcija baca ValueError kada ulazni signal nema dovoljno uzoraka
    za zadani broj simbola
    """
    NFFT = 4
    NCP = 2
    NSYM = NFFT + NCP
    num_symbols = 3
    start = 0
    rx = np.arange(NSYM*num_symbols - 1) + 1j*np.arange(NSYM*num_symbols - 1)  # premalo
    
    with pytest.raises(ValueError):
        remove_cp(rx, start, num_symbols, NSYM, NCP, NFFT)

@pytest.mark.parametrize("param_name, val", [
    ("num_symbols", -1),
    ("NSYM", -5),
    ("NCP", -2),
    ("NFFT", -8)
])
def test_remove_cp_negative_values(param_name, val):
    """
    Provjerava da funkcija baca ValueError kada se predaju negativne vrijednosti za
    ključne parametre (num_symbols, NSYM, NCP, NFFT)
    """
    NFFT = 8
    NCP = 2
    NSYM = NFFT + NCP
    num_symbols = 1
    start = 0
    rx = np.arange(NSYM*num_symbols) + 1j*np.arange(NSYM*num_symbols)
    
    kwargs = dict(num_symbols=num_symbols, NSYM=NSYM, NCP=NCP, NFFT=NFFT)
    kwargs[param_name] = val
    
    with pytest.raises(ValueError):
        remove_cp(rx, start, **kwargs)

def test_remove_cp_invalid_types():
    """
    Provjerava da funkcija baca TypeError kada ulazni parametri nisu odgovarajućeg tipa:
    - rx nije array
    - num_symbols nije int
    - NSYM nije int
    """
    NFFT = 8
    NCP = 2
    NSYM = NFFT+NCP
    num_symbols = 1
    start = 0
    rx = "not an array"
    
    with pytest.raises(TypeError):
        remove_cp(rx, start, num_symbols, NSYM, NCP, NFFT)
    
    rx = np.arange(NSYM) + 1j*np.arange(NSYM)
    with pytest.raises(TypeError):
        remove_cp(rx, start, "one", NSYM, NCP, NFFT)
    with pytest.raises(TypeError):
        remove_cp(rx, start, num_symbols, 3.5, NCP, NFFT)

def test_remove_cp_single_symbol():
    """
    Provjerava da funkcija ispravno radi za jedan simbol:
    - uklanja CP
    - vraća FFT
    """
    NFFT = 4
    NCP = 1
    NSYM = NFFT + NCP  # 5
    num_symbols = 1
    start = 0
    # niz dovoljno dug da sadrži cijeli simbol sa CP
    rx = np.arange(NSYM*num_symbols) + 1j*np.arange(NSYM*num_symbols)
    
    symbols_fd = remove_cp(rx, start, num_symbols, NSYM, NFFT, NCP)
    
    print("symbols_fd shape:", symbols_fd.shape)
    assert symbols_fd.shape == (num_symbols, NFFT)

def test_remove_cp_invalid_nsym():
    """
    Provjerava greške za neispravne NSYM vrijednosti:
    - NCP + NFFT > NSYM
    - NSYM nije int
    """
    NFFT = 4
    NCP = 16
    NSYM = 10  
    num_symbols = 1
    start = 0
    rx = np.arange(NSYM*num_symbols) + 1j*np.arange(NSYM*num_symbols)
    
    # Provjera da baca ValueError zbog NCP + NFFT > NSYM
    with pytest.raises(ValueError, match="NCP \\+ NFFT ne smije biti veće od NSYM"):
        remove_cp(rx, start, num_symbols, NSYM, NFFT, NCP)

    NSYM_float = 80.0
    with pytest.raises(TypeError, match="NSYM mora biti int tipa"):
        remove_cp(rx, start, num_symbols, NSYM_float, NFFT, NCP)