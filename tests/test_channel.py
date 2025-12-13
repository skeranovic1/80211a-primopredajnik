import pytest
import numpy as np
from channel.channel_mode import ChannelMode
from channel.channel_settings import ChannelSettings
from channel.Channel_Model import  Channel_Model
from channel.Multipath import GetMultipathFilter
from channel.AWGN import Generate_AWGN
import scipy.signal as sc

# ===========================
# TESTOVI ZA ChannelMode
# ===========================

def test_channelmode_defaults():
    mode = ChannelMode()
    assert mode.Multipath == 0
    assert mode.ThermalNoise == 1

def test_channelmode_set_valid():
    mode = ChannelMode()
    mode.Multipath = 1
    mode.ThermalNoise = 0
    assert mode.Multipath == 1
    assert mode.ThermalNoise == 0

def test_channelmode_set_invalid():
    mode = ChannelMode()
    with pytest.raises(ValueError):
        mode.Multipath = 2
    with pytest.raises(ValueError):
        mode.ThermalNoise = -1

# ===========================
# TESTOVI ZA ChannelSettings
# ===========================

def test_channelsettings_defaults():
    settings = ChannelSettings()
    assert settings.SampleRate == 40e6
    assert settings.NumberOfTaps == 40
    assert settings.DelaySpread == 150e-9
    assert settings.SNR_dB == 35

def test_channelsettings_set_valid():
    settings = ChannelSettings()
    settings.SampleRate = 1e6
    settings.NumberOfTaps = 10
    settings.DelaySpread = 1e-6
    settings.SNR_dB = 20
    assert settings.SampleRate == 1e6
    assert settings.NumberOfTaps == 10
    assert settings.DelaySpread == 1e-6
    assert settings.SNR_dB == 20

def test_channelsettings_set_invalid():
    settings = ChannelSettings()
    with pytest.raises(ValueError):
        settings.SampleRate = 0
    with pytest.raises(ValueError):
        settings.NumberOfTaps = -5
    with pytest.raises(ValueError):
        settings.DelaySpread = -1

# ===========================
# TESTOVI ZA Channel_Model
# ===========================

def test_channel_model_apply_multipath_awgn():
    settings = ChannelSettings(number_of_taps=5, delay_spread=50e-9, snr_db=20)
    mode = ChannelMode(multipath=1, thermal_noise=1)
    model = Channel_Model(settings, mode)
    
    tx_samples = np.array([1.0, 0.5, -1.0, 0.2, -0.5])
    out_samples, fir_taps = model.apply(tx_samples)
    out_samples = np.ravel(out_samples)
    # Provjera da FIR filter ima ispravan broj tapova
    assert len(fir_taps) == settings.NumberOfTaps
    
    # Provjera da izlaz nije identičan ulazu (multipath + AWGN mijenja signal)
    assert not np.allclose(out_samples, tx_samples)
    
    # Provjera da izlaz ima istu dužinu kao ulaz
    assert len(out_samples) == len(tx_samples)

def test_channel_model_apply_no_multipath_no_awgn():
    settings = ChannelSettings(number_of_taps=5, delay_spread=50e-9, snr_db=20)
    mode = ChannelMode(multipath=0, thermal_noise=0)
    model = Channel_Model(settings, mode)
    
    tx_samples = np.array([1.0, 2.0, 3.0])
    out_samples, fir_taps = model.apply(tx_samples)
    out_samples = np.ravel(out_samples)  # <-- dodaj ovu liniju
    
    # Kada je multipath i AWGN isključen, izlaz bi trebao biti samo normalizovan
    expected = tx_samples / np.sqrt(np.var(tx_samples))
    np.testing.assert_array_almost_equal(out_samples, expected)
