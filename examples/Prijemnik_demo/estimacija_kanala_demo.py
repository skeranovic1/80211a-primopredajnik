import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from rx.pretprocessing import iq_preprocessing
from rx.detection import packet_detector
from rx.cfo import detect_frequency_offsets, gruba_vremenska_sinhronizacija
from rx.estimacija_kanala import channel_estimate_and_equalizer
from rx.PhaseCorrection_80211a import phase_correction
from rx.rastavljanje import remove_cp

num_ofdm_symbols=500
up_factor=2
fs_base=20e6
fs=fs_base*up_factor

#Predajnik
tx=Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   #QPSK
        step=1,
        up_factor=up_factor,
        seed=4,
        plot=False
)
tx_signal, tx_bits, _= tx.generate_frame()

#Kanal
settings=ChannelSettings(
        sample_rate=fs,
        number_of_taps=2,
        delay_spread=10e-9,
        snr_db=25
)
mode=ChannelMode(
        multipath=1,
        thermal_noise=1
)
channel=Channel_Model(settings, mode)
rx_signal, fir_taps=channel.apply(tx_signal)

#Prijemnik - pretprocesing, detekcija i korekcija
rx_signal, fs1=iq_preprocessing(
        rx_signal=rx_signal,
        tx_signal=tx_signal,
        fs=fs
)

_, _, start_lts, _ = packet_detector(rx_signal)
print("Kraj detektovane short training sekvence:", start_lts)
print(f"Očekivani kraj STS-a (sample): {160}")
print(f"Greska pri detekciji STS-a: {160-start_lts} uzoraka")
    
#Symbol timing
end_lts = start_lts + 160  #32 CP + 2x64 korisna
rx_lts = rx_signal[start_lts:end_lts]
pravi_pocetak_lts, _, _ = gruba_vremenska_sinhronizacija(rx_lts, search_win=32)
#ovdje pravi pocetak LTS-a znaci precizno odredivanje pozicije (ima male vrijednosti, ~10 uzoraka), 
# jer se posmatra samo LTS dio primljenog signala. Slijedi postavljanje na globalne pozicije

#FFT start u globalnim indeksima
lts_start = start_lts + pravi_pocetak_lts 
ideal_lts_start = 160+32 #pozicioniranje na pocetak korisnog dijela LTS-a, poslije CP 
timing_error = ideal_lts_start  - lts_start
print("Detektovani pocetak korisnog dijela LTS-a:", lts_start)
print("Idealni pocetak korisnog dijela LTS-a:", ideal_lts_start)
print(f"Symbol timing greška: {timing_error} uzoraka ({timing_error/fs1*1e6:.2f} µs)")

#Detekcija frekvencijskih ofseta
#Coarse/gruba korekcija
FreqOffset=detect_frequency_offsets(rx_signal,lts_start,fs1) #LTS start je pocetak korisnog dijela, bez CP
CoarseOffset=FreqOffset[0]
print(f"Coarse CFO = {CoarseOffset:.2f} Hz")
n=np.arange(len(rx_signal))
NCO_coarse=np.exp(-1j*2*np.pi*n*CoarseOffset/fs1) #korekcija
rx_coarse=rx_signal*NCO_coarse

# Fine/precizna korekcija
FreqOffset=detect_frequency_offsets(rx_coarse,lts_start,fs1)  #ponovo se pokreće detekcija za fine offset
FineOffset=FreqOffset[1]
NCO_fine=np.exp(-1j*2*np.pi*n*FineOffset/fs1) #korekcija 
rx_fine=rx_coarse*NCO_fine
print(f"Fine CFO = {FineOffset:.3f} Hz")

#Skidanje CP-a 
NFFT = 64          
NCP = 16          
NSYM = NFFT + NCP  #80 uzoraka po OFDM simbolu
data_start = lts_start + 2 * 64 #pocetak podataka nakon LTS-a, lts_start + duzina dva LTS simbola
print("Pocetak podataka (sample):", data_start)
symbols_fd = remove_cp(rx_fine,data_start,num_ofdm_symbols, NSYM, NFFT, NCP)

plt.scatter(
    symbols_fd.real.flatten(),
    symbols_fd.imag.flatten(),
    s=3
)
plt.axis("equal")
plt.grid(True)
plt.title("Konstelacija nakon CP skidanja")
plt.show()

#Estimacija kanala i koeficijenti equalizera
samo_lts=rx_fine[lts_start : data_start] #koristi se samo korisni dio LTS-a, ovaj put poslije korekcije
channel_est, eq_coefficient = channel_estimate_and_equalizer(samo_lts)
ekvalizirani_simboli= symbols_fd * eq_coefficient #korekcija
H = np.fft.fft(fir_taps, NFFT) #za impulsni odziv 

#Fazna korekcija
corrected_symbols = phase_correction(ekvalizirani_simboli,num_ofdm_symbols,channel_est)
phase_before = np.unwrap(np.angle(ekvalizirani_simboli)) #Faza prije fazne korekcije za poredenje
phase_after = np.unwrap(np.angle(corrected_symbols))  #Faza poslije fazne korekcije za poredenje

#Konstelacija svih OFDM simbola prije i poslije korekcije
plt.figure(figsize=(6,6))
plt.scatter(ekvalizirani_simboli.real.flatten(), ekvalizirani_simboli.imag.flatten(), s=3, label='Prije korekcije', alpha=0.5)
plt.scatter(corrected_symbols.real.flatten(), corrected_symbols.imag.flatten(), s=3, label='Poslije korekcije', alpha=0.5)
plt.axis('equal')
plt.title(f'Konstelacija svih OFDM simbola ({num_ofdm_symbols} simbola)')
plt.grid(True)
plt.legend()
plt.show()

#Spektar podnosioca prije i poslije korekcije
plt.figure(figsize=(10,4))
plt.plot(np.fft.fftshift(np.abs(ekvalizirani_simboli[0])), label='Prije korekcije')
plt.plot(np.fft.fftshift(np.abs(corrected_symbols[0])), label='Poslije korekcije')
plt.title('Spektar podnosača OFDM simbola (prvi simbol)')
plt.xlabel('Subcarrier index')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.show()

#Konstelacija prije i poslije korekcije
plt.figure(figsize=(6,6))
plt.scatter(ekvalizirani_simboli[0].real, ekvalizirani_simboli[0].imag, s=5, label='Prije korekcije')
plt.scatter(corrected_symbols[0].real, corrected_symbols[0].imag, s=5, label='Poslije korekcije')
plt.axis('equal')
plt.title('Konstelacija OFDM simbola (prvi simbol)')
plt.grid(True)
plt.legend()
plt.show()

#Kanal i equalizer
plt.figure(figsize=(12,4))
plt.plot(np.abs(channel_est), 'o-', label='Channel Estimate |H(f)|')
plt.plot(np.abs(eq_coefficient), 'x-', label='Equalizer |1/H(f)|')
plt.title('Channel and Equalizer')
plt.xlabel('Subcarrier Index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.show()

"""def Demapper_OFDM(symbols, bits_per_symbol):

    # LUT tabele (identične tvojim iz mappera)
    BPSK_LUT  = np.array([-1, 1])
    QPSK_LUT  = np.array([-1, 1]) / np.sqrt(2)
    QAM16_LUT = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    QAM64_LUT = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)

    all_bits = []
    
    for s in symbols:
        if bits_per_symbol == 1:  # BPSK
            # Tražimo najbliži indeks (0 ili 1) u BPSK_LUT
            idx = np.argmin(np.abs(BPSK_LUT - np.real(s)))
            all_bits.append(idx)
            
        elif bits_per_symbol == 2:  # QPSK
            idx_i = np.argmin(np.abs(QPSK_LUT - np.real(s)))
            idx_q = np.argmin(np.abs(QPSK_LUT - np.imag(s)))
            all_bits.extend([idx_i, idx_q])
            
        elif bits_per_symbol == 4:  # 16-QAM
            idx_i = np.argmin(np.abs(QAM16_LUT - np.real(s)))
            idx_q = np.argmin(np.abs(QAM16_LUT - np.imag(s)))
            # Binarna dekompozicija indeksa (obrnuto od: bg[0]*2 + bg[1])
            all_bits.extend([(idx_i >> 1) & 1, idx_i & 1, 
                             (idx_q >> 1) & 1, idx_q & 1])
            
        elif bits_per_symbol == 6:  # 64-QAM
            idx_i = np.argmin(np.abs(QAM64_LUT - np.real(s)))
            idx_q = np.argmin(np.abs(QAM64_LUT - np.imag(s)))
            # Binarna dekompozicija (obrnuto od: bg[0]*4 + bg[1]*2 + bg[2])
            all_bits.extend([(idx_i >> 2) & 1, (idx_i >> 1) & 1, idx_i & 1,
                             (idx_q >> 2) & 1, (idx_q >> 1) & 1, idx_q & 1])
    
    return np.array(all_bits)

#Demapiramo bite iz flatten niza corrected_symbols 
rx_bits = Demapper_OFDM(corrected_symbols.flatten(), tx.bits_per_symbol)
tx_bits_compare = tx_bits[:len(rx_bits)]

plt.figure(figsize=(10, 5))
error_distribution = (rx_bits != tx_bits_compare).astype(int)

plt.stem(error_distribution, markerfmt='o', basefmt="r-") 
plt.title("Distribucija grešaka (1 = pogrešno, 0 = ispravno)")
plt.xlabel("Bit index")
plt.ylabel("Vrijednost greške")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

plt.figure(figsize=(10, 5))
ber_per_symbol = []
bits_in_one_sym = 48 * tx.bits_per_symbol 

for i in range(num_ofdm_symbols):
    start = i * bits_in_one_sym
    end = (i + 1) * bits_in_one_sym
    sym_tx = tx_bits_compare[start:end]
    sym_rx = rx_bits[start:end]
    
    if len(sym_tx) > 0:
        errs = np.sum(sym_tx != sym_rx)
        ber_per_symbol.append(errs / len(sym_tx))

plt.bar(range(len(ber_per_symbol)), ber_per_symbol, width=1.0)
plt.title(f"Stabilnost prenošenja: BER po simbolu")
plt.xlabel("OFDM simbol index")
plt.ylabel("Bit Error Rate (BER)")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()

final_err_count = np.sum(rx_bits != tx_bits_compare)
print("-" * 35)
print(f"Ukupno obrađenih bita: {len(rx_bits)}")
print(f"Broj pogrešnih bita:   {final_err_count}")
print(f"Konačni BER sistema:   {final_err_count / len(rx_bits):.6f}")"""