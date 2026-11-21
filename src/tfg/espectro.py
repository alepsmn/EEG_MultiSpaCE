import numpy  as np

def espectro(x, fs):
    """Devuelve (freqs, mag).
    
    frqs: frecuencia en Hz de cada bin.
    mag: magnitud normalizada tal que un seno de amplitud A
    produce mag[bin de su frecuencia] == A
    """
    imagin_coef = np.fft.rfft(x)
    mag = np.abs(imagin_coef)
    norm_mag = mag * (2 / len(x)) # len(x) = N
    freqs = np.fft.rfftfreq(len(x), 1/fs)

    return freqs, norm_mag

if __name__ == "__main__":
    fs = 256
    f = 10 
    T = 8
    # df = 1 / T
    N = fs * T # 2048
    t = np.arange(N) / fs
    A = 1
    sen = A*np.sin(2*np.pi*f*t)

    freqs, norm_mag = espectro(sen, fs)

    print(freqs, norm_mag, len(freqs))

    assert len(freqs) == 1025, f"longitud no esperada: 1025 vs {len(freqs)}"
    df = np.diff(freqs)
    assert np.all(np.isclose(df, 0.125)), f"df no cumplio con lo esperado: 0.125 vs {df}"
    assert np.argmax(norm_mag) == 80, f"Posicion del pico incorrecta -{np.argmax(norm_mag)} - esperado: 80"
    assert freqs[np.argmax(norm_mag)] == 10.0, f"Valor de frecuencia esperado (10.0) incorrecto {freqs[np.argmax(norm_mag)]}"
    assert norm_mag[80] == 1.0, f"Valor de amplitud esperado (1) - incorrecto {norm_mag[80]}" # -> mejor np.isclose()
