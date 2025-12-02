# 802.11a OFDM Transmitter (Work in Progress)

## Status
Ovaj projekt trenutno implementira osnovne dijelove OFDM TX lanca:

- OFDM mapper za BPSK, QPSK, 16-QAM i 64-QAM
- Generisanje short i long training sequences
- IFFT sa guard intervalom (GI)
- Upsampling i half-band filtriranje
- Osnovni testovi pokrivenosti OFDM mappera

**Trenutno nije implementirano:**

- Model kanala
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

- `from tx.OFDM_TX_802_11 import OFDM_TX_802_11`  
- Generiše 5 OFDM simbola sa QPSK modulacijom (2 bita po simbolu):  
  `samples, symbols = OFDM_TX_802_11(NumberOf_OFDM_Symbols=5, BitsPerSymbol=2)`  
  `print("Oblik signala:", samples.shape)`  
  `print("Prikaz simbola:", symbols)`

## Testiranje

Testovi koriste `pytest` i pokrivaju trenutno:

- OFDM mapper  
- Half-band upsampling filter  
- Zero-stuffing funkcije  

Pokretanje testiranja:  
`pytest`

## Struktura projekta


- `tx/` — Glavni kod  
- `tests/` — Pytest testovi  
- `README.md`  
- `requirements.txt`



## Plan razvoja / TODO

- Dodati RX lanac i sinhronizaciju  
- Poboljšati testove za integraciju cijelog sistema  
- Dodati dokumentaciju i primjere korištenja
