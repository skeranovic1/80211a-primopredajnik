import numpy as np

def packet_detector(rx_input):
    """
   
    """

    rx_input = np.asarray(rx_input, dtype=np.complex128)
    N = len(rx_input)

    # Outputs
    autocorr_est = np.zeros(N, dtype=np.complex128)
    comparison_ratio = np.zeros(N)
    packet_det_flag = np.zeros(N, dtype=int)
    falling_edge_position = None

    #Unutrasnja stanja
    delay16 = np.zeros(16, dtype=np.complex128)
    sliding_avg_autocorr = np.zeros(32, dtype=np.complex128)
    sliding_avg_power = np.zeros(32)
    detection_flag = 0

    for i in range(N):

        #16-uzoraka kasnjenje
        rx_delayed = delay16[-1]
        delay16[1:] = delay16[:-1]
        delay16[0] = rx_input[i]

        #Estimacija autokorelacije
        temp = rx_input[i] * np.conj(rx_delayed)
        sliding_avg_autocorr[1:] = sliding_avg_autocorr[:-1]
        sliding_avg_autocorr[0] = temp

        autocorr_est[i] = np.sum(sliding_avg_autocorr) / 32
        abs_autocorr_est = np.abs(autocorr_est[i])

        #Estimacija varijanse
        inst_power = rx_input[i] * np.conj(rx_input[i])
        sliding_avg_power[1:] = sliding_avg_power[:-1]
        sliding_avg_power[0] = inst_power.real

        variance_est = np.sum(sliding_avg_power) / 32

        #Poredenje
        if variance_est > 0:
            comparison_ratio[i] = abs_autocorr_est / variance_est
        else:
            comparison_ratio[i] = 0.0

        #Detekcija paketa sa histerezom
        if comparison_ratio[i] > 0.85:
            detection_flag = 1
        elif comparison_ratio[i] < 0.65:
            detection_flag = 0

        packet_det_flag[i] = detection_flag

        #Falling edge detekcija
        if (
            i > 0
            and i<N/2
            and packet_det_flag[i] - packet_det_flag[i - 1] == -1
        ):
            falling_edge_position = i

    return comparison_ratio, packet_det_flag, falling_edge_position, autocorr_est
