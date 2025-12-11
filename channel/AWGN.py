import numpy as np

def Generate_AWGN(Input, SNR, sd=0):
    np.random.seed(sd) 
    MeanSquare = (1 / len(Input)) * (Input @ Input.conj().T)
    # Total signal power
    NoisePower = MeanSquare / (10 ** (SNR / 10))  
    STDNoise = np.sqrt(NoisePower) 
    l = 1  
    Noise = STDNoise * (0.70711 * np.random.randn(l, len(Input)) + 1j * 0.70711 * np.random.randn(l, len(Input)))
    return Noise