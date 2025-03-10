# EE329 Signals and Systems 2
# Project from Ch02
# 2025-03-08



#################################################
# IMPORT MODULES
#################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math



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
f_0 = 1 / T_0



#################################################
# PART 2
#################################################
'''
k_range = np.arange(-10, 11)  # k from -10 to 10
k_range_nonzero = k_range[k_range != 0]  # Exclude k = 0 to avoid division by zero

# Fourier coefficients (compute b_k for nonzero k)
# Alternate signs using simple conditional logic
signs = np.where(k_range_nonzero % 2 == 0, 1, -1)  # Assign +1 for even k, -1 for odd k
b_k = -1 * a / (np.pi * k_range_nonzero) * signs  # sine coefficients (b_k)
c_k = np.zeros_like(k_range_nonzero)  # cosine coefficients (c_k), zero for sawtooth

# Magnitude Spectrum: |ak| = sqrt(c_k^2 + b_k^2)
ak_magnitude = np.sqrt(c_k**2 + b_k**2)

# Phase Spectrum: phase(ka) = atan2(b_k, c_k)
ak_phase = np.angle(b_k + 1j*c_k)  # Using complex number for easy phase computation

# Flip the phase for negative k values to point downwards
ak_phase[k_range_nonzero < 0] *= -1

# Plotting
plt.figure(figsize=(12, 6))

# Magnitude Spectrum with height labels
plt.subplot(1, 2, 1)
markerline, stemlines, baseline = plt.stem(k_range_nonzero, ak_magnitude, basefmt=" ", linefmt='b-', markerfmt='bo')  # Use k_range_nonzero for x-axis
plt.setp(stemlines, 'linewidth', 1)

for i, mag in enumerate(ak_magnitude):
    plt.text(k_range_nonzero[i], mag + 0.1, f'{mag:.2f}', ha='center', va='bottom', fontsize=8)

plt.axvline(x=0, color='b', linestyle='-', ymax=5/2.5)
plt.ylim(0, 3)
plt.xlabel('f_0 k')  # Change x-axis label back to k
plt.ylabel('Magnitude |a_k|')
plt.title('Magnitude Spectrum')
plt.grid()

# Phase Spectrum with height labels
plt.subplot(1, 2, 2)
markerline, stemlines, baseline = plt.stem(k_range_nonzero, ak_phase, basefmt=" ", linefmt='b-', markerfmt='bo')  # Use k_range_nonzero for x-axis
plt.setp(stemlines, 'linewidth', 1)

for i, phase in enumerate(ak_phase):
    plt.text(k_range_nonzero[i], phase + 0.1, f'{phase:.2f}', ha='center', va='bottom', fontsize=8)

plt.xlabel('f_0 k')  # Change x-axis label back to k
plt.ylabel('Phase a_k (radians)')
plt.title('Phase Spectrum')
plt.grid()

plt.tight_layout()
'''


#################################################
# PART 2
#################################################

# calculations
mag_of_ak = []
phase_of_ak = []
k_values = list(range(-10, 10))  # x-axis values

for i in k_values:
    # magnitude
    if i == 0:
        mag_of_ak.append(1000)  # its infinity but this is tall enough for the plot
    else:
        mag_of_ak.append(4 / (math.pi * abs(i)))

    # phase
    if i == 0:
        phase_of_ak.append(0.0)
    elif i > 0:
        phase_of_ak.append(math.pi)
    else:
        phase_of_ak.append(-math.pi) 

# create figure
plt.figure(figsize=(12, 5))  # adjust figure size to make plots more readable

# magnitude plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
markerline, stemline, baseline = plt.stem(k_values, mag_of_ak, markerfmt="bo") 
baseline.set_visible(False)
plt.xlabel("f_0 k")
plt.ylabel("|a_k|")
plt.ylim(0, 3)
plt.title("Magnitude of Fourier Series Coefficients")
plt.grid(True)

for x, y in zip(k_values, mag_of_ak):
    plt.text(x, y + 0.1, f"{y:.2f}", ha="center", fontsize=6)

# phase plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
markerline, stemline, baseline = plt.stem(k_values, phase_of_ak, markerfmt="bo")
baseline.set_visible(False)
plt.xlabel("f_0 k")
plt.ylabel("Phase of a_k (radians)")
plt.ylim(-4, 4) 
plt.title("Phase of Fourier Series Coefficients")
plt.grid(True)

for x, y in zip(k_values, phase_of_ak):
    plt.text(x, y + 0.2, f"{y:.2f}", ha="center", fontsize=6)

# plot
plt.tight_layout()  # adjusts spacing to prevent overlap
plt.show()




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
plt.xlim(0, 4000)  # zoom in on specific range
plt.title("Fourier Transform Magnitude Spectrum")

# plot narrowed freq spectrum
plt.figure()
plt.plot(f[:len(f)//2], np.abs(FTdata[:len(f)//2]))  # only positive frequencies
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(15, 25)  # zoom in on specific range
plt.axhline(y=10000, color='grey', linestyle='--', alpha=0.5)
plt.title("Fourier Transform Magnitude Spectrum")



#################################################
# PLOT
#################################################

# show plots for everything above
plt.show()
