# 802.11a primopredajnik

## Status
Ovaj projekt trenutno implementira osnovne dijelove OFDM TX lanca:

- OFDM mapper za BPSK, QPSK, 16-QAM i 64-QAM
- Generisanje short i long training sequences
- IFFT sa guard intervalom (GI)
- Upsampling i half-band filtriranje
- Model kanala (AWGN i multipath)
- Sinhronizacija prema short i long training sekvencama
- Vremenska i frekventna sinhronizacija
- Skidanje cikličnog prefiksa
- Kanalna kompenzacija i fazna korekcija
- Testovi pokrivenosti trenutno implementiranih funkcija

## Instalacija

1. Kloniraj repozitorij i uđi u direktorij:  
   `git clone https://github.com/skeranovic1/80211a-primopredajnik.git`  
   `cd 80211a-primopredajnik`

2. Kreiraj i aktiviraj virtualno okruženje:  
   `python -m venv .venv`  
   Na Linux/macOS: `source .venv/bin/activate`  
   Na Windows: `.venv\Scripts\activate`

3. Instaliraj zavisnosti:  
   `pip install -r requirements.txt`

## Korištenje trenutno implementiranih funkcija

U Python skripti ili interaktivnom okruženju:  

### Predajnik
- `from tx.OFDM_TX_802_11 import OFDM_TX_802_11`  
- Generiše 5 OFDM simbola sa QPSK modulacijom (2 bita po simbolu):  
  `samples, symbols = OFDM_TX_802_11(NumberOf_OFDM_Symbols=5, BitsPerSymbol=2)`  
  `print("Oblik signala:", samples.shape)`  
  `print("Prikaz simbola:", symbols)`

### Kanal
- `from channel.Channel_Model import Channel_Model`  
- Inicijalizacija kanala sa željenim parametrima i modom:  
  `chan = Channel_Model(settings, mode)`  
- Primjena kanala na OFDM uzorke:  
  `tx_samples_channel, fir_taps = chan.apply(samples)`  
  `print("Oblik signala nakon kanala:", tx_samples_channel.shape`  
  `print("FIR taps:", fir_taps)`

### Prijemnik
- `from rx.RX_802_11_a import receiver80211a`  
- Procesira primljeni signal nakon prolaska kroz kanal i generiše obrađene simbole uz pomoć metode `process_signal()`: 
  `corrected_symbols = rx.process_signal(rx_signal, tx_signal)`
  `print("Oblik obrađenih simbola:", corrected_symbols.shape)`
  `print("Prikaz obrađenih simbola:", corrected_symbols)`  
  
## Testiranje

Testovi koriste `pytest` i pokrivaju trenutno:

- OFDM mapper (`test_OFDM_mapper.py`)  
- Half-band upsampling filter (`test_half_band_upsample.py`)  
- Zero-stuffing i utilities funkcije (`test_utilities.py`)  
- Generisanje i obrada OFDM simbola (`test_ifft_ofdm_symbol.py`, `test_ifft_gi.py`)  
- Short i long training sekvence (`test_short_sequence.py`, `test_long_sequence.py`)  
- Predajnini paket (`test_predajnikt.py`)  
- Model kanala, uključujući AWGN i multipath kanale (`test_channel.py`, `test_awgn_channel.py`, `test_multipath_channel.py`)  
- Pretprocesiranje i detekcija paketa (`test_pretprocessing.py`, `test_detection.py`)
- LTS korelator (`test_long_symbol_correlator.py`)
- Vremenska i frekventna sinhronizacija (`test_offsets.py`)
- Izvlačenje data podnosioca (`test_removecp.py`)
- Estimacija kanala (`test_estimacija.py`)
- Fazna korekcija (`test_phasecorrection.py`)
- Prijemnik (`test_prijemnik.py`)

Pokretanje testiranja:  
`pytest`

## Struktura projekta

- `tx/` — 802.11a OFDM predajni lanac  
- `channel/` — Model kanala (AWGN i multipath)
- `rx/` —  802.11a OFDM prijemni lanac
- `gui/` — Grafički korisnički interfejs za podešavanje i vizualizaciju  
- `examples/` — Primjeri korištenja  
- `tests/` — Testovi  
- `README.md` — Projektna dokumentacija  
- `requirements.txt` — Python zavisnosti  
- `setup.py` — Setup skripta

## Dokumentacija

- Automatski generisana Doxygen HTML dokumentacija dostupna je ovdje:  
   https://skeranovic1.github.io/80211a-primopredajnik/

## Proširenje sistema
- Implementacija dodatnih blokova u predajnom i prijemnom lancu radi povećanja robusnosti i realističnije simulacije sistema.

- Predajnik (Tx lanac):
  - Scrambler: Uklanja duge nizove istih bita i poboljšava spektralne karakteristike signala
  - Kanalno kodiranje: Dodavanje redundantnih bita radi povećanja otpornosti na greške
  - Interleaver: Raspoređuje bite kako bi se smanjio uticaj burst grešaka prije modulacije

- Prijemnik (Rx lanac):
  - Deinterleaver: Vraća bite u originalni redoslijed nakon prijema
  - Dekoder: Rekonstruiše originalni niz podataka iz primljenih simbola koristeći Viterbi ili Turbo dekodiranje
  - Descrambler: Vraća originalne podatke uklanjanjem efekta scrambler-a

- Analiza performansi:
  - Poređenje performansi sistema sa i bez dodatnih blokova
  - Evaluacija uticaja kodiranja i interleaving-a na grešku bita (BER)
