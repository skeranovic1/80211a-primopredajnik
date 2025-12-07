import matplotlib.pyplot as plt
import numpy as np 
from short_sequence import get_short_training_sequence
from long_sequence import get_long_training_sequence

def plot_sequences(step=1):
    """
    Prikazuje realni i imaginarni dio obje sekvence (short i long) u vremenskom domenu.
    """
    short_seq = get_short_training_sequence(step)
    long_seq = get_long_training_sequence(step)

    plt.figure(figsize=(12,6))

    plt.subplot(2,2,1)
    plt.title("Short Sequence - Realni dio")
    plt.plot(short_seq.real)
    plt.grid()

    plt.subplot(2,2,2)
    plt.title("Short Sequence - Imaginarni dio")
    plt.plot(short_seq.imag)
    plt.grid()

    plt.subplot(2,2,3)
    plt.title("Long Sequence - Realni dio")
    plt.plot(long_seq.real)
    plt.grid()

    plt.subplot(2,2,4)
    plt.title("Long Sequence - Imaginarni dio")
    plt.plot(long_seq.imag)
    plt.grid()

    plt.xlim(0,160)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_sequences(step=1)
