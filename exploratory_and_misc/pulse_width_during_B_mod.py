import sys
sys.path.insert(1, '../')  # Add parent directory to path for custom libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from scipy.optimize import curve_fit

# Custom libraries
from analysis.library import styles, plt_settings, colors
from analysis.breit_rabi import FreqMHz, Ehf


plt_settings['font.size'] = 16
plt_settings["legend.fontsize"] = 12
plt.rcParams.update(plt_settings)

def f97(B):
    """Frequency of the 9/2,-9/2 to 9/2,-7/2 transition in kHz at field B (in Gauss)."""
    return FreqMHz(B, 9/2, -9/2, 9/2, -7/2)*1e3  # in kHz


def field_modulation(t, B0, Bmod, fmod):
    """Calculate the magnetic field with modulation."""
    return B0 + Bmod * np.sin(2 * np.pi * fmod * t)


def square_pulse(t, pulse_width, t0):
    """Generate a square pulse."""
    return np.where((t >= t0 - pulse_width / 2) & (t <= t0 + pulse_width / 2), 1.0, 0.0)


def sinc2(detuning, pulse_width):
    return (np.sinc(detuning * pulse_width))**2


def freq_shift(dB, B0=202.14):
    """Convert magnetic field to frequency shift in kHz."""
    return (f97(dB + B0) - f97(B0))  # in kHz


def B_shift(df, B0=202.14):
    """Convert frequency shift in kHz to magnetic field in Gauss."""
    dB = 0.1  # Small field step in Gauss
    return df * dB / (f97(B0 + dB) - f97(B0 - dB))  # in Gauss


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
    return np.trapezoid(sinc2(detunings, pulse_width), dx=t[1]-t[0])


def percentile_range(x_vals, y_vals, percentiles=(40, 60)):
    """Calculate the range between the given percentiles."""
    lower, upper = percentiles
    return np.percentile(x_vals, upper, method='inverted_cdf', weights=y_vals) \
            - np.percentile(x_vals, lower, method='inverted_cdf', weights=y_vals)

def fit_to_sinc2(x, y):
    """Fit data to a sinc^2 function."""
    guess_freq_width = 1/0.1  # 1/ms
    popt, _ = curve_fit(lambda x, x0, pw: sinc2(x-x0, 1/pw), x, y, p0=[0, guess_freq_width])
    return popt[0]*2, popt[1]


###
### Parameters
###
B0 = 0  # arb.
Bmod = 0.1  # arb.
fmod = 10  # kHz
tmin = 0  # ms
tmax = 0.2  # ms
t = np.linspace(tmin, tmax, 1000)  # from 0 to 0.2 ms
dB = field_modulation(t, B0, Bmod, fmod)
hatch_patterns = ['/', '\\', '|']
delta = 0.001  # Small offset to avoid overlap in hatch patterns

def pulse_times(tmin, tmax, num_pulses=5):
    """Generate center times for a series of pulses."""
    return np.linspace(tmin + 0.1/fmod, tmax - 0.1/fmod, num_pulses)

pulse_widths = [0.01, 0.02, 0.04, 0.08, 0.1]  # ms

fit_B_mods = []
for pulse_width in pulse_widths:

    ###
    ### Plot field response vs. time and pulse widths.
    ###
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2,2, figure=fig, height_ratios=[3,1])

    ### Plot response vs. detuning
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set(xlabel='Detuning (kHz)', ylabel='Normalized Response (arb.)')

    pulse_centres = pulse_times(tmin, tmax, num_pulses=7)  # Center times of pulses.
    pulse_amp = min(pulse_widths)/pulse_width  # Normalize pulse amplitude for visibility

    patch_width = pulse_width - delta # Adjusted width for visibility

    detunings = np.linspace(-2.5/pulse_width, 2.5/pulse_width, 2000)  # kHz
    ax0.set(xlim=(detunings[0], detunings[-1]))
    static_response = sinc2(detunings, pulse_width)
    ax0.plot(detunings, static_response/np.max(static_response), '--', color='k',
                label='static')

    spectra_widths = []
    B_responses = []
    for time, color in zip(pulse_centres, colors):
        dBt = lambda t: field_modulation(t, B0, Bmod, fmod)  # Bfield as function of time.

        response_vals = np.array([response(d, time, dBt, pulse_width) for d in detunings])

        normalized_response = response_vals/np.max(response_vals)
        percentile_width = B_shift(percentile_range(detunings, normalized_response)/2)

        shift, width = fit_to_sinc2(detunings, normalized_response)
        B_responses.append(B_shift(shift))
        spectra_widths.append(B_shift(width))

        ax0.plot(detunings, normalized_response, '-', color=color,
                    label='t={:.2f} ms'.format(time))
    ax0.legend()

    popt, pcov = curve_fit(lambda t, Bmod: Bmod*np.sin(2*np.pi*t*fmod), pulse_centres, B_responses, p0=[Bmod])
    perr = np.sqrt(np.diag(pcov))
    print("Fitted parameters: Bmod={:.4f}({:.0f})".format(popt[0], perr[0]*1e4))
    fit_B_mods.append(popt[0])
    fit_B_oscillation = lambda t: popt[0]*np.sin(2*np.pi*t*fmod)

    ### Plot Field modulation and pulse sequence
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    ax1.set(xlim=(tmin, tmax), ylabel='B field change (arb.)', xticklabels=[])
    ax2.set(xlim=(tmin, tmax), xlabel='Time (ms)')

    # Plot the magnetic field modulation
    ax1.plot(t, fit_B_oscillation(t), '-', label=r'fit $\delta B(t)$')
    ax1.plot(t, dB, '--', label=r'$\delta B(t)$')

    for i, time in enumerate(pulse_centres):
        # dB_at_pulse = field_modulation(time, B0, Bmod, fmod)
        
        xerr = pulse_width/2
        ax1.errorbar(time, B_responses[i], #yerr=spectra_widths[i], xerr=xerr, 
                    fmt='o', color=colors[i], label='t={:.2f} ms'.format(time))

        rect = patches.Rectangle((time - patch_width/2, 0), patch_width, pulse_amp, 
                                linewidth=1, edgecolor=colors[i], facecolor=colors[i], 
                                alpha=0.5, hatch=hatch_patterns[0])
        ax2.add_patch(rect)

    # ax1.legend()

    fig.suptitle("Pulse width = {:.2f} ms, Modulation period = {:.2f} ms".format(pulse_width, 1/fmod))
    fig.tight_layout()
    plt.show()


plt.figure(figsize=(6,4))
plt.plot(np.array([0] + pulse_widths)*fmod, np.array([0.1] + fit_B_mods)/Bmod, 'o-')
plt.xlabel('Pulse width/Period')
plt.ylabel('Fit B amplitude/B amp')
plt.tight_layout()
# plt.title("Non-interacting, transfer during field modulation",)
plt.show()

