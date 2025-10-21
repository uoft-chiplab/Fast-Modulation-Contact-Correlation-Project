import matplotlib.patches as patches

# enable custom imports from analysis directory 
from preamble import *

# Custom libraries
from library import plt_settings, colors
from breit_rabi import FreqMHz


plt_settings['font.size'] = 16
plt_settings["legend.fontsize"] = 12
plt.rcParams.update(plt_settings)


def linear(x, m, c):
    return m*x + c


# Contact measurements from not sure
Cs = [0.44, 0.78, 1.27]
Bs = [202.3, 202.1, 201.9]

c_popt, pcov = curve_fit(linear, Bs, Cs)
c_perr = np.sqrt(np.diag(pcov))


def contact_response(dB):
    """Contact response function approximation."""
    return linear(202.14 + dB, *c_popt)


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
    Function to calculate system response to an rf pulse with detuning applied at time
    t0 during a magnetic field modulation Bt with defined pulse width.

    Parameters:
    detuning : float
        Detuning of the rf pulse in kHz.
    t : float
        Time corresponding to centre of pulse, in ms.
    Bt : function
        Magnetic field as a function of time in ms.
    pulse_width : float
        Width of the pulse in ms.
    num : int
        Number of points in integration array.

    Returns: float
        The calculated response.
    """
    t = np.linspace(t0 - pulse_width/2, t0 + pulse_width/2, num)  # Time array around pulse center
    detunings = detuning*np.ones(num) - freq_shift(Bt(t))
    return np.trapezoid(contact_response(Bt(t))*sinc2(detunings, pulse_width), dx=t[1]-t[0])


def percentile_range(x_vals, y_vals, percentiles=(40, 60)):
    """Calculate the range between the given percentiles."""
    lower, upper = percentiles
    return np.percentile(x_vals, upper, method='inverted_cdf', weights=y_vals) \
            - np.percentile(x_vals, lower, method='inverted_cdf', weights=y_vals)

def fit_to_sinc2(x, y, pulse_width):
    """Fit data to a sinc^2 function."""
    guess_freq_width = 1/pulse_width  # 1/ms
    popt, _ = curve_fit(lambda x, A, x0, pw: A*sinc2(x-x0, 1/pw), x, y, 
                        p0=[1/guess_freq_width, 0, guess_freq_width])
    return popt[1]*2, popt[2], popt[0]  # return shift, width, amplitude


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

# pulse_widths = [0.01]

fit_B_mods = []
fit_B_offs = []
fit_B_phases = []
for pulse_width in pulse_widths:

    ###
    ### Plot field response vs. time and pulse widths.
    ###
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2,2, figure=fig, height_ratios=[3,1])

    ### Plot response vs. detuning
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.set(xlabel='Detuning (kHz)', ylabel='Normalized Response (arb.)')

    pulse_centres = pulse_times(tmin, tmax, num_pulses=9)  # Center times of pulses.
    pulse_amp = min(pulse_widths)/pulse_width  # Normalize pulse amplitude for visibility

    patch_width = pulse_width - delta # Adjusted width for visibility

    # detunings = np.linspace(-2.5/pulse_width, 2.5/pulse_width, 2000)  # kHz
    detunings = np.linspace(-5/pulse_width, 5/pulse_width, 2000)  # kHz
    ax0.set(xlim=(detunings[0], detunings[-1]))
    static_response = sinc2(detunings, pulse_width)
    static_norm = np.trapezoid(static_response, dx=detunings[1]-detunings[0])
    ax0.plot(detunings, static_response/static_norm, '--', color='k',
                label='static')

    spectra_widths = []
    B_responses = []
    for time, color in zip(pulse_centres, colors):
        dBt = lambda t: field_modulation(t, B0, Bmod, fmod)  # Bfield as function of time.

        response_vals = np.array([response(d, time, dBt, pulse_width) for d in detunings])
        response_norm = np.trapezoid(response_vals, dx=detunings[1]-detunings[0])
        normalized_response = response_vals/response_norm

        percentile_width = B_shift(percentile_range(detunings, normalized_response)/2)

        shift, width, amp = fit_to_sinc2(detunings, normalized_response, pulse_width)
        B_responses.append(B_shift(shift))
        spectra_widths.append(B_shift(width))

        ax0.plot(detunings, normalized_response, '-', color=color,
                    label='t={:.2f} ms'.format(time))
    ax0.legend()

    popt, pcov = curve_fit(lambda t, Bmod, Boff, phi: Bmod*np.sin(2*np.pi*t*fmod + phi) + Boff, 
                           pulse_centres, B_responses, p0=[Bmod, 0, 0])
    perr = np.sqrt(np.diag(pcov))
    print("Fitted parameters: Bmod={:.4f}({:.0f}), Boff={:.4f}({:.0f}), phi={:.4f}({:.0f})".format(popt[0], 
                                            perr[0]*1e4, popt[1], perr[1]*1e4, popt[2], perr[2]*1e4))
    fit_B_mods.append(popt[0])
    fit_B_offs.append(popt[1])
    fit_B_phases.append(popt[2])
    fit_B_oscillation = lambda t: popt[0]*np.sin(2*np.pi*t*fmod + popt[2]) + popt[1]

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
