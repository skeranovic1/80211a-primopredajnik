import numpy as np

def Channel_Model(TX_Samples, Settings, Mode):
    # 1. Transferring the mode parameters
    Multipath_Select = Mode.Multipath  # 0/1 means exclude/include 
    AW_GaussianNoise_Select = Mode.ThermalNoise

    # 2. Transferring settings
    N = Settings.NumberOfTaps  # For Multipath Modelpy
    DelaySpread = Settings.DelaySpread
    SampleRate = Settings.SampleRate
    SNR_dB = Settings.SNR_dB  # Thermal Noise

    # 3. Generating multipath filterpp
    FIR_Taps = GetMultipathFilter(SampleRate, DelaySpread, N)
    if Multipath_Select == 1:
        TX_Samples = lfilter(FIR_Taps, 1, TX_Samples)
    VarOutput = np.var(TX_Samples)
    TX_Samples = TX_Samples / np.sqrt(VarOutput)

    # 4. Generating additive white gaussian noise (thermal noise)
    if AW_GaussianNoise_Select == 1:
        TX_Samples = TX_Samples + Generate_AWGN(TX_Samples, SNR_dB)

    return TX_Samples, FIR_Taps
