# EE329 Signals and Systems 2
# Project from Ch02
# 2025-03-08



#################################################
# IMPORT MODULES
#################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft



#################################################
# PART 4
#################################################

# assigned parameters
a = 4
b = 2

# defining parameters
amplitude = a
t_0 = -b
T_0 = 2 * b
step = 0.01
t = np.arange((-T_0 / 2), (T_0 / 2), step)

# original inverse sawtooth function for reference
x_t = a * (t / (T_0 / 2))  # x(t) = -4 * (t / 2) over [-2,2]
plt.figure(figsize=(6, 4))
plt.plot(t, x_t, color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Assigned Inverse Sawtooth Function")
plt.grid()

# fourier series approximation function
def fourier_series_inverse_sawtooth(t, N, T_0, a):
    x_approx = np.zeros_like(t)
    b_0 = 0  # DC component, for sawtooth this is zero
    for k in range(1, N + 1):
        b_k = -2 * a / (np.pi * k) * (-1)**k  # note that c_k = 0 for sawtooth
        x_approx += b_k * np.sin(2 * np.pi * k * t / T_0) 
    return x_approx

# fourier approximations and residual signals
N_values = [1, 3, 5, 15, 100]

# results and plots
for N in N_values:

    # fourier approximations
    x_approx = fourier_series_inverse_sawtooth(t, N, T_0, a)
    plt.figure(figsize=(6, 4))
    plt.plot(t, x_t, linestyle="dashed", color="blue") # original for reference
    plt.plot(t, x_approx, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Fourier Approximation (N={N})")
    plt.grid()
    
    # compute and plot residual signals
    x_approx = fourier_series_inverse_sawtooth(t, N, T_0, a)
    residual = x_t - x_approx
    plt.figure(figsize=(6, 4))
    plt.plot(t, residual, color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Residual Amplitude")
    plt.title(f"Residual Signal (N = {N})")
    plt.grid()

    # compute residual power
    x_approx = fourier_series_inverse_sawtooth(t, N, T_0, a)
    residual = x_t - x_approx
    P_res = (1 / T_0) * np.sum(residual**2) * step
    print(f"Residual Power for N = {N}: {P_res:.4f}") 



#################################################
# PART 5
#################################################

# load data
time = np.loadtxt("time.txt")
intensity = np.loadtxt("intensity.txt")

# apply FFT
FTdata = fft(intensity)
df = 1 / (time[-1] - time[0])
f = np.fft.fftfreq(len(time), d=(time[1] - time[0]))

# plot time domain signal
plt.figure()
plt.plot(time, intensity)
plt.xlabel("Time (s)")
plt.ylabel("Intensity")
plt.title("Time-Domain Signal")

# plot frequency spectrum
plt.figure()
plt.plot(f[:len(f)//2], np.abs(FTdata[:len(f)//2]))  # only positive frequencies
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(5, 15)  # zoom in on specific range
plt.axhline(y=10000, color='grey', linestyle='--', alpha=0.5)
plt.title("Fourier Transform Magnitude Spectrum")



#################################################
# PLOT
#################################################

# show plots for everything above
plt.show()
