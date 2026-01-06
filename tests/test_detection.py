import numpy as np
from rx.detection import packet_detector  

def test_noise_only():
    """Test 1: Čisti šum: ne smije detektovati paket"""
    np.random.seed(0)
    N = 2000
    rx_noise = (np.random.randn(N) + 1j*np.random.randn(N))/np.sqrt(2)
    
    cr, flag, fe, ac = packet_detector(rx_noise)
    
    assert np.sum(flag) == 0
    assert fe is None

def test_ideal_sts():
    """Test 2: Idealni STS – paket mora biti detektovan"""
    np.random.seed(0)
    sts_base = np.exp(1j*2*np.pi*np.random.rand(16))
    sts = np.tile(sts_base, 10)
    
    noise = 0.1*(np.random.randn(len(sts)) + 1j*np.random.randn(len(sts)))
    rx_sts = sts + noise
    
    rx = np.concatenate([
        (np.random.randn(300)+1j*np.random.randn(300))/np.sqrt(2),
        rx_sts,
        (np.random.randn(300)+1j*np.random.randn(300))/np.sqrt(2)
    ])
    
    cr, flag, fe, ac = packet_detector(rx)
    
    assert np.sum(flag) > 0
    assert fe is not None
  

def test_cfo_robustness():
    """Test 3: STS sa frekventnim offsetom (CFO)"""
    np.random.seed(0)
    sts_base = np.exp(1j*2*np.pi*np.random.rand(16))
    sts = np.tile(sts_base, 10)
    
    cfo = 100e3
    fs = 20e6
    n = np.arange(len(sts))
    rx_sts_cfo = sts * np.exp(1j*2*np.pi*cfo*n/fs)
    
    rx = np.concatenate([
        (np.random.randn(300)+1j*np.random.randn(300))/np.sqrt(2),
        rx_sts_cfo,
        (np.random.randn(300)+1j*np.random.randn(300))/np.sqrt(2)
    ])
    
    cr, flag, fe, ac = packet_detector(rx)
    
    assert np.sum(flag) > 0
    assert fe is not None
    

def test_short_signal():
    """Test 4: Kratki signal: rubni slučaj"""
    np.random.seed(0)
    rx_short = (np.random.randn(20)+1j*np.random.randn(20))/np.sqrt(2)
    
    cr, flag, fe, ac = packet_detector(rx_short)
    
    assert np.sum(flag) == 0
    assert fe is None

