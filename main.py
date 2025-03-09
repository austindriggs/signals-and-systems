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
# PARAMETERS
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



#################################################
# PART 2
#################################################

k_range = np.arange(-10, 11)  # k from -10 to 10
k_range_nonzero = k_range[k_range != 0]  # Exclude k = 0 to avoid division by zero

# Fourier coefficients (compute b_k for nonzero k)
# Alternate signs using simple conditional logic
signs = np.where(k_range_nonzero % 2 == 0, 1, -1)  # Assign +1 for even k, -1 for odd k
b_k = -2 * a / (np.pi * k_range_nonzero) * signs  # sine coefficients (b_k)
c_k = np.zeros_like(k_range_nonzero)  # cosine coefficients (c_k), zero for sawtooth

# Magnitude Spectrum: |ka| = sqrt(c_k^2 + b_k^2)
ka_magnitude = np.sqrt(c_k**2 + b_k**2)

# Phase Spectrum: phase(ka) = atan2(b_k, c_k)
ka_phase = np.angle(b_k + 1j*c_k)  # Using complex number for easy phase computation

# Plotting
plt.figure(figsize=(12, 6))

# Magnitude Spectrum
plt.subplot(1, 2, 1)
plt.stem(k_range_nonzero, ka_magnitude, basefmt=" ", linefmt='b-', markerfmt='bo')  # solid blue line and blue markers
plt.axvline(x=0, color='b', linestyle='-', ymax=5/2.5)  # Solid blue line at x=0
plt.ylim(0, 3)  # Set y-limits for the magnitude spectrum
plt.xlabel('k (Discrete Frequencies)')
plt.ylabel('Magnitude |ka|')
plt.title('Magnitude Spectrum')
plt.grid()

# Phase Spectrum
plt.subplot(1, 2, 2)
plt.stem(k_range_nonzero, ka_phase, basefmt=" ", linefmt='b-', markerfmt='bo')  # solid blue line and blue markers
plt.xlabel('k (Discrete Frequencies)')
plt.ylabel('Phase (ka) [radians]')
plt.title('Phase Spectrum')
plt.grid()

plt.tight_layout()



#################################################
# PART 4
#################################################

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
