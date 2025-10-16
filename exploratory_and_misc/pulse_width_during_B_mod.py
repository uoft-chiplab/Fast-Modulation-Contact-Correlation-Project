import sys
sys.path.insert(1, '../')  # Add parent directory to path for custom libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Custom libraries
from analysis.library import styles, plt_settings, colors
from analysis.breit_rabi import FreqMHz

plt.rcParams.update(plt_settings)

def f97(B):
    """Frequency of the 9/2,-9/2 to 9/2,-7/2 transition in kHz at field B (in Gauss)."""
    return FreqMHz(B, 9/2, -9/2, 9/2, -7/2)*1e3  # in kHz

def field_modulation(t, B0, Bmod, fmod):
    """
    Calculate the magnetic field with modulation.

    Parameters:
    B0 : float
        The static magnetic field in Tesla.
    Bmod : float
        The amplitude of the modulation in Tesla.
    fmod : float
        The frequency of the modulation in Hz.
    t : array-like
        Time array in seconds.

    Returns:
    B_total : array-like
        The total magnetic field at each time point.
    """
    return B0 + Bmod * np.sin(2 * np.pi * fmod * t)

def pulse_times(tmin, tmax, pulse_width, num_pulses=5):
    """
    Generate center times for a series of pulses.

    Parameters:
    tmin : float
        Minimum time in ms.
    tmax : float
        Maximum time in ms.
    pulse_width : float
        Width of each pulse in ms.
    num_pulses : int
        Number of pulses.

    Returns:
    pulse_times : array-like
        Center times of the pulses in ms.
    """
    return np.linspace(tmin + pulse_width / 2, tmax - pulse_width / 2, num_pulses)


def square_pulse(t, pulse_width, t0):
    """
    Generate a square pulse.

    Parameters:
    t : array-like
        Time array in ms.
    pulse_width : float
        Width of the pulse in ms.
    t0 : float
        Center time of the pulse in ms.

    Returns:
    pulse : array-like
        The amplitude of the pulse at each time point.
    """
    return np.where((t >= t0 - pulse_width / 2) & (t <= t0 + pulse_width / 2), 1.0, 0.0)

def sinc2(detuning, pulse_width):
    """Parameters:
    detuning : array-like
        Frequency detuning array in kHz.
    pulse_width : float
        Width of the pulse in ms.

    Returns:
    sinc2_values : array-like
        The sinc^2 values at each frequency detuning.
    """
    return (np.sinc(detuning * pulse_width))**2

def freq_shift(B, B0=202.14):
    """
    Convert magnetic field to frequency shift in kHz.

    Parameters:
    B : array-like
        Magnetic field in Gauss
    B0 : float
        Reference field in Guass, default is 202.14 G.

    Returns:
    freq : array-like
        Frequency shift in kHz.
    """
    return (f97(B) - f97(B0))  # in kHz

def response(detuning, t0, Bt, pulse_width, num=1000):
    """
    Function to calculate system response to a pulse at magnetic field B.
    Replace with actual physics-based calculation as needed.

    Parameters:
    t : float
        Time corresponding to centre of pulse, in ms.
    Bt : function
        Magnetic field as a function of time in ms.
    pulse_width : float
        Width of the pulse in ms.
    num : int
        Number of points in integration array.

    Returns:
    response : float
        The calculated response.
    """
    t = np.linspace(t0 - pulse_width/2, t0 + pulse_width/2, num)  # Time array around pulse center
    detunings = detuning*np.ones(num) - freq_shift(Bt(t))
    return np.trapezoid(sinc2(detunings, pulse_width), dx=t[1]-t[0])/num

###
### Parameters
###
B0 = 0  # arb.
Bmod = 1  # arb.
fmod = 10  # kHz
tmin = 0  # ms
tmax = 0.2  # ms
t = np.linspace(tmin, tmax, 1000)  # from 0 to 0.2 ms
B_total = field_modulation(t, B0, Bmod, fmod)

pulse_widths = [0.02, 0.04, 0.08]  # ms
hatch_patterns = ['/', '\\', '|']
delta = 0.001  # Small offset to avoid overlap in hatch patterns


###
### Plot response vs. detuning and pulse widths
###
fig, ax = plt.subplots(figsize=(6, 4))
ax.set(xlabel='Detuning (kHz)', ylabel='Response (arb.)')

pulse_centres = pulse_times(tmin, tmax, min(pulse_widths))  # Center times of pulses

for time, pulse_width, color in zip(pulse_centres, pulse_widths, colors):
    detunings = np.linspace(-2/pulse_width, 2/pulse_width, 200)  # kHz
    ax.plot(detunings, sinc2(detunings, pulse_width), '--', color=color,
            label='static response')
    
    response_vals = [response(d, time, lambda t: field_modulation(t, B0, Bmod, fmod), pulse_width) for d in detunings]
    ax.plot(detunings, response_vals, '-', color=color,
            label='response at t={:.2f} ms'.format(time))
    break

ax.legend()
fig.tight_layout()
plt.show()

###
### Plot field response vs. time and pulse widths
###
fig, axs = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])

axs[0].set_ylabel('Magnetic Field (arb.)')
axs[1].set_ylabel('Pulse Amp (arb.)')

for ax in axs:
    ax.set(xlim=(tmin, tmax), xlabel='Time (ms)')

# Plot the magnetic field modulation
ax = axs[0]
ax.plot(t, B_total, '--', label='B(t)')

for pulse_width, color, hatch_pattern in zip(pulse_widths, colors, hatch_patterns):
    pulse_centres = pulse_times(tmin, tmax, pulse_width) # Center times of pulses
    B_at_pulses = field_modulation(pulse_centres, B0, Bmod, fmod)

    axs[0].plot(pulse_centres, B_at_pulses, 'o', color=color, 
                label='w={:.2f} ms'.format(pulse_width))

    # Plot the rf pulses as rectangles
    for pt in pulse_centres:
        pulse_amp = min(pulse_widths)/pulse_width  # Normalize pulse amplitude for visibility
        width = pulse_width - delta # Adjusted width for visibility
        rect = patches.Rectangle((pt - width/2, 0), width, pulse_amp, 
                                linewidth=1, edgecolor=color, facecolor=color, alpha=0.5,
                                hatch=hatch_pattern)
        axs[1].add_patch(rect)

axs[0].legend()

fig.tight_layout()
plt.show()


