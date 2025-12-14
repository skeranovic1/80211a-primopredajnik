# 802.11a OFDM Transmitter (Work in Progress)

## Status
Ovaj projekt trenutno implementira osnovne dijelove OFDM TX lanca:

- OFDM mapper za BPSK, QPSK, 16-QAM i 64-QAM
- Generisanje short i long training sequences
- IFFT sa guard intervalom (GI)
- Upsampling i half-band filtriranje
- Model kanala (AWGN i multipath)
- Testovi pokrivenosti trenutno implementiranih funkcija

**Trenutno nije implementirano:**

- RX lanac i sinkronizacija

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
  `print("Oblik signala nakon kanala:", tx_samples_channel.shape)`  
  `print("FIR taps:", fir_taps)`

## Testiranje

Testovi koriste `pytest` i pokrivaju trenutno:

- OFDM mapper (`test_OFDM_mapper.py`)  
- Half-band upsampling filter (`test_half_band_upsample.py`)  
- Zero-stuffing i utilities funkcije (`test_utilities.py`)  
- Generisanje i obrada OFDM simbola (`test_ifft_ofdm_symbol.py`, `test_ifft_gi.py`)  
- Short i long training sekvence (`test_short_sequence.py`, `test_long_sequence.py`)  
- Predajnini paket (`test_tx_packet.py`)  
- Model kanala, uključujući AWGN i multipath kanale (`test_channel.py`, `test_awgn_channel.py`, `test_multipath_channel.py`)  

Pokretanje testiranja:  
`pytest`

## Struktura projekta

- `tx/` — 802.11a OFDM predajnički lanac  
- `channel/` — Model kanala (AWGN i multipath)  
- `gui/` — Grafički korisnički interfejs za podešavanje i vizualizaciju  
- `examples/` — Primjeri korištenja  
- `tests/` — Automatski testovi  
- `README.md` — Projektna dokumentacija  
- `requirements.txt` — Python zavisnosti  
- `setup.py` — Setup skripta

## Dokumentacija

- Automatski generisana Doxygen HTML dokumentacija dostupna je ovdje:  
   [text](https://skeranovic1.github.io/80211a-primopredajnik/)

## Plan razvoja / TODO

- Dodati RX lanac i sinhronizaciju  
- Poboljšati testove za integraciju cijelog sistema  
- Dodati dokumentaciju i primjere korištenja
