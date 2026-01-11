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
from rx.PhaseCorrection_80211a import phase_correction_80211a
from rx.rastavljanje import remove_cp
from scipy.signal import freqz

num_ofdm_symbols=120
up_factor=2
fs_base=20e6
fs=fs_base*up_factor

#Predajnik
tx=Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   #QPSK
        step=1,
        up_factor=up_factor,
        seed=3,
        plot=False
)
tx_signal,_, pocetni_simboli= tx.generate_frame()

#Kanal
settings=ChannelSettings(
        sample_rate=fs,
        number_of_taps=40,
        delay_spread=10e-9,
        snr_db=10
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

#Estimacija kanala i koeficijenti equalizera
samo_lts=rx_fine[lts_start : data_start] #koristi se samo korisni dio LTS-a, ovaj put poslije korekcije
channel_est, eq_coefficient = channel_estimate_and_equalizer(samo_lts)
ekvalizirani_simboli= symbols_fd * eq_coefficient #korekcija
H = np.fft.fft(fir_taps, NFFT) #za impulsni odziv 

#Fazna korekcija
corrected_symbols = phase_correction_80211a(ekvalizirani_simboli,num_ofdm_symbols,channel_est)
phase_before = np.unwrap(np.angle(ekvalizirani_simboli)) #Faza prije fazne korekcije za poredenje
phase_after = np.unwrap(np.angle(corrected_symbols))  #Faza poslije fazne korekcije za poredenje

#Performance Evaluation (EVM)
corrected_symbols=corrected_symbols.flatten() #jer je corrected_symbols kao matrica, ovo ga postavlja kao array
ErrorVectors = pocetni_simboli-corrected_symbols
Average_ErrorVectorPower = np.mean(np.abs(ErrorVectors) ** 2)

EVM_dB = 10 * np.log10(Average_ErrorVectorPower)
print(f"EVM = {EVM_dB:.2f} dB")

# EVM vs Time
Error_Time = np.zeros(num_ofdm_symbols)

for i in range(num_ofdm_symbols):
    s = i * 48
    e = s + 48
    ev = tx_signal[s:e] - corrected_symbols[s:e]
    Error_Time[i] = np.mean(np.abs(ev) ** 2)

EVM_Time_dB = 10 * np.log10(Error_Time)

# EVM vs Frequency
Error_Frequency = np.zeros(48)

for i in range(num_ofdm_symbols):
    s = i * 48
    e = s + 48
    ev = tx_signal[s:e] - corrected_symbols[s:e]
    Error_Frequency += np.abs(ev) ** 2 / num_ofdm_symbols

EVM_Frequency_dB = 10 * np.log10(Error_Frequency)

f = np.arange(-0.5, 0.501, 0.001) 
Response = np.zeros(len(f), dtype=complex)
n = np.arange(len(fir_taps))

for d in range(len(f)):
    E = np.exp(1j * 2 * np.pi * n * f[d])
    Response[d] = np.dot(fir_taps, np.conj(E))

MagResponse = 20 * np.log10(np.abs(Response) + 1e-12)
MagResponse_norm = MagResponse - np.max(MagResponse)

#Plot rjesenja
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(EVM_Frequency_dB, 'k.')
plt.title("EVM Versus Frequency")
plt.xlabel("Tones")
plt.ylabel("dB")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(range(1, num_ofdm_symbols + 1), EVM_Time_dB, 'k')
plt.title("EVM Versus Time")
plt.xlabel("Symbols")
plt.ylabel("dB")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(f, MagResponse_norm, 'k')
plt.title("Magnitude Response of Multipath Filter")
plt.xlabel("Frequency")
plt.ylabel("dB")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.stem(np.abs(fir_taps), linefmt='k-', markerfmt='ko', basefmt='k-')
plt.title("FIR Taps")
plt.xlabel("Symbols")
plt.grid(True)

plt.tight_layout()
plt.show()

