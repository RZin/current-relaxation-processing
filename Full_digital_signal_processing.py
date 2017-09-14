"""This file  extended for exp_dec_gauss_tau analysis
"""

from __future__ import print_function, division

from collections import OrderedDict

import array
import copy
import math

import numpy as np
import random
import scipy
import scipy.stats
import scipy.fftpack

import subprocess
import thinkplot
import warnings
import pandas as pd

from wave import open as open_wave

try:
    from IPython.display import Audio
except:
    warnings.warn("Can't import Audio from IPython.display; "
                  "Wave.make_audio() will not work.")

# written for additional functions
import math as m
import matplotlib.pyplot as plt

PI2 = math.pi * 2
screen = (20, 15)

def random_seed(x):
    """Initialize the random and np.random generators.

    x: int seed
    """
    random.seed(x)
    np.random.seed(x)


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


class WavFileWriter:
    """Writes wav files."""

    def __init__(self, filename='sound.wav', framerate=11025):
        """Opens the file and sets parameters.

        filename: string
        framerate: samples per second
        """
        self.filename = filename
        self.framerate = framerate
        self.nchannels = 1
        self.sampwidth = 2
        self.bits = self.sampwidth * 8
        self.bound = 2 ** (self.bits - 1) - 1

        self.fmt = 'h'
        self.dtype = np.int16

        self.fp = open_wave(self.filename, 'w')
        self.fp.setnchannels(self.nchannels)
        self.fp.setsampwidth(self.sampwidth)
        self.fp.setframerate(self.framerate)

    def write(self, wave):
        """Writes a wave.

        wave: Wave
        """
        zs = wave.quantize(self.bound, self.dtype)
        self.fp.writeframes(zs.tostring())

    def close(self, duration=0):
        """Closes the file.

        duration: how many seconds of silence to append
        """
        if duration:
            self.write(rest(duration))

        self.fp.close()


def read_wave(filename='sound.wav'):
    """Reads a wave file.

    filename: string

    returns: Wave
    """
    fp = open_wave(filename, 'r')

    nchannels = fp.getnchannels()
    nframes = fp.getnframes()
    sampwidth = fp.getsampwidth()
    framerate = fp.getframerate()

    z_str = fp.readframes(nframes)

    fp.close()

    dtype_map = {1: np.int8, 2: np.int16, 3: 'special', 4: np.int32}
    if sampwidth not in dtype_map:
        raise ValueError('sampwidth %d unknown' % sampwidth)

    if sampwidth == 3:
        xs = np.fromstring(z_str, dtype=np.int8).astype(np.int32)
        ys = (xs[2::3] * 256 + xs[1::3]) * 256 + xs[0::3]
    else:
        ys = np.fromstring(z_str, dtype=dtype_map[sampwidth])

    # if it's in stereo, just pull out the first channel
    if nchannels == 2:
        ys = ys[::2]

    # ts = np.arange(len(ys)) / framerate
    wave = Wave(ys, framerate=framerate)
    wave.normalize()
    return wave


def play_wave(filename='sound.wav', player='aplay'):
    """Plays a wave file.

    filename: string
    player: string name of executable that plays wav files
    """
    cmd = '%s %s' % (player, filename)
    popen = subprocess.Popen(cmd, shell=True)
    popen.communicate()


def find_index(x, xs):
    """Find the index corresponding to a given value in an array."""
    n = len(xs)
    start = xs[0]
    end = xs[-1]
    i = round((n - 1) * (x - start) / (end - start))
    return int(i)


class _SpectrumParent:
    """Contains code common to Spectrum and DCT.
    """

    def __init__(self, hs, fs, framerate, full=False):
        """Initializes a spectrum.

        hs: array of amplitudes (real or complex)
        fs: array of frequencies
        framerate: frames per second
        full: boolean to indicate full or real FFT
        """
        self.hs = np.asanyarray(hs)
        self.fs = np.asanyarray(fs)
        self.framerate = framerate
        self.full = full

    def get_fs(self):
        return self.fs

    def get_hs(self):
        return self.hs

    @property
    def max_freq(self):
        """Returns the Nyquist frequency for this spectrum."""
        return self.framerate / 2

    @property
    def amps(self):
        """Returns a sequence of amplitudes (read-only property)."""
        return np.absolute(self.hs)

    @property
    def power(self):
        """Returns a sequence of powers (read-only property)."""
        return self.amps ** 2

    def copy(self):
        """Makes a copy.

        Returns: new Spectrum
        """
        return copy.deepcopy(self)

    def max_diff(self, other):
        """Computes the maximum absolute difference between spectra.

        other: Spectrum

        returns: float
        """
        assert self.framerate == other.framerate
        assert len(self) == len(other)

        hs = self.hs - other.hs
        return np.max(np.abs(hs))

    def ratio(self, denom, thresh=1, val=0):
        """The ratio of two spectrums.

        denom: Spectrum
        thresh: values smaller than this are replaced
        val: with this value

        returns: new Wave
        """
        ratio_spectrum = self.copy()
        ratio_spectrum.hs /= denom.hs
        ratio_spectrum.hs[denom.amps < thresh] = val
        return ratio_spectrum

    def invert(self):
        """Inverts this spectrum/filter.

        returns: new Wave
        """
        inverse = self.copy()
        inverse.hs = 1 / inverse.hs
        return inverse

    @property
    def freq_res(self):
        return self.framerate / 2 / (len(self.fs) - 1)

    def render_full(self, high=None):
        """Extracts amps and fs from a full spectrum.

        high: cutoff frequency

        returns: fs, amps
        """
        hs = np.fft.fftshift(self.hs)
        amps = np.abs(hs)
        fs = np.fft.fftshift(self.fs)
        i = 0 if high is None else find_index(-high, fs)
        j = None if high is None else find_index(high, fs) + 1
        return fs[i:j], amps[i:j]

    def plot(self, high=None, **options):
        """Plots amplitude vs frequency.

        Note: if this is a full spectrum, it ignores low and high

        high: frequency to cut off at
        """
        if self.full:
            fs, amps = self.render_full(high)
            thinkplot.plot(fs, amps, **options)
        else:
            i = None if high is None else find_index(high, self.fs)
            thinkplot.plot(self.fs[:i], self.amps[:i], **options)

    def smart_plot(self, high=None, logx=False):
        if self.full:
            fs, amps = self.render_full(high)
            plt.plot(fs, amps)
            plt.xscale = 'log'
        else:
            i = None if high is None else find_index(high, self.fs)
            plt.figure(figsize=screen)
            plt.plot(self.fs[:i], self.amps[:i])
            plt.xscale = 'log'

    def plot_power(self, high=None, **options):
        """Plots power vs frequency.

        high: frequency to cut off at
        """
        if self.full:
            fs, amps = self.render_full(high)
            thinkplot.plot(fs, amps ** 2, **options)
        else:
            i = None if high is None else find_index(high, self.fs)
            thinkplot.plot(self.fs[:i], self.power[:i], **options)

    def estimate_slope(self):
        """Runs linear regression on log power vs log frequency.

        returns: slope, inter, r2, p, stderr
        """
        x = np.log(self.fs[1:])
        y = np.log(self.power[1:])
        t = scipy.stats.linregress(x, y)
        return t

    def peaks(self):
        """Finds the highest peaks and their frequencies.

        returns: sorted list of (amplitude, frequency) pairs
        """
        t = list(zip(self.amps, self.fs))
        t.sort(reverse=True)
        return t


class Spectrum(_SpectrumParent):
    """Represents the spectrum of a signal."""
    def make_wave(self):
        """Transforms to the time domain.

        returns: Wave
        """
        if self.full:
            ys = np.fft.ifft(self.hs)
        else:
            ys = np.fft.irfft(self.hs)

        # NOTE: whatever the start time was, we lose it when
        # we transform back; we could fix that by saving start
        # time in the Spectrum
        # ts = self.start + np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=self.framerate)

    def make_dataframe(self):
        # create pandas dataframe
        datum = OrderedDict()
        datum['Frequency'] = self.fs
        datum['Complex'] = self.hs
        datum['Real'] = self.real
        datum['Imaginary'] = self.imag
        datum['Magnitude'] = self.amps*len(self.hs)
        datum['Amplitude'] = self.amps
        datum['Phase'] = self.angles
        datum['framerate'] = self.framerate
        df = pd.DataFrame(datum)
        # print(df)
        return df

    def save_spectrum_txt(self, filename):
        '''creates txt Spectrum_+filename and puts dataframe of time and vals
        '''
        '''creates txt file only from ys and ts'''
        # create pandas dataframe
        df = self.make_dataframe()
        path = r'C:\Python\_FFT_fitting_project\spectrum_data\\'
        df.to_csv(path + filename, header=True, index=False, sep='\t', mode='w')
        return


    def __len__(self):
        """Length of the spectrum."""
        return len(self.hs)

    def __add__(self, other):
        """Adds two spectrums elementwise.

        other: Spectrum

        returns: new Spectrum
        """
        if other == 0:
            return self.copy()

        assert all(self.fs == other.fs)
        hs = self.hs + other.hs
        return Spectrum(hs, self.fs, self.framerate, self.full)

    __radd__ = __add__

    def __mul__(self, other):
        """Multiplies two spectrums elementwise.

        other: Spectrum

        returns: new Spectrum
        """
        assert all(self.fs == other.fs)
        hs = self.hs * other.hs
        return Spectrum(hs, self.fs, self.framerate, self.full)

    def convolve(self, other):
        """Convolves two Spectrums.

        other: Spectrum

        returns: Spectrum
        """
        assert all(self.fs == other.fs)
        if self.full:
            hs1 = np.fft.fftshift(self.hs)
            hs2 = np.fft.fftshift(other.hs)
            hs = np.convolve(hs1, hs2, mode='same')
            hs = np.fft.ifftshift(hs)
        else:
            # not sure this branch would mean very much
            hs = np.convolve(self.hs, other.hs, mode='same')

        return Spectrum(hs, self.fs, self.framerate, self.full)

    @property
    def real(self):
        """Returns the real part of the hs (read-only property)."""
        return np.real(self.hs)

    @property
    def imag(self):
        """Returns the imaginary part of the hs (read-only property)."""
        return np.imag(self.hs)

    @property
    def angles(self):
        """Returns a sequence of angles (read-only property)."""
        return np.angle(self.hs)

    def scale(self, factor):
        """Multiplies all elements by the given factor.

        factor: what to multiply the magnitude by (could be complex)
        """
        self.hs *= factor

    def low_pass(self, cutoff, factor=0):
        """Attenuate frequencies above the cutoff.

        cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        self.hs[abs(self.fs) > cutoff] *= factor

    def high_pass(self, cutoff, factor=0):
        """Attenuate frequencies below the cutoff.

        cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        self.hs[abs(self.fs) < cutoff] *= factor

    def band_stop(self, low_cutoff, high_cutoff, factor=0):
        """Attenuate frequencies between the cutoffs.

        low_cutoff: frequency in Hz
        high_cutoff: frequency in Hz
        factor: what to multiply the magnitude by
        """
        # TODO: test this function
        fs = abs(self.fs)
        indices = (low_cutoff < fs) & (fs < high_cutoff)
        self.hs[indices] *= factor

    def pink_filter(self, beta=1):
        """Apply a filter that would make white noise pink.

        beta: exponent of the pink noise
        """
        denom = self.fs ** (beta / 2.0)
        denom[0] = 1
        self.hs /= denom

    def differentiate(self):
        """Apply the differentiation filter.

        returns: new Spectrum
        """
        new = self.copy()
        new.hs *= PI2 * 1j * new.fs
        return new

    def integrate(self):
        """Apply the integration filter.

        returns: new Spectrum
        """
        new = self.copy()
        new.hs /= PI2 * 1j * new.fs
        return new

    def make_integrated_spectrum(self):
        """Makes an integrated spectrum.
        """
        cs = np.cumsum(self.power)
        cs /= cs[-1]
        return IntegratedSpectrum(cs, self.fs)

class ExpDecSpectrum(Spectrum):
    def __init__(self, y0, amp, tau, sig, Nexps, xmin, xmax, npoints, \
                 hs, fs, framerate, full=False):
        """Initializes the ExpDecWave. Same as other waves but with additional parameters
        ys: wave array
        ts: array of times
        framerate: samples per second
        y0, amp, tau, sig, Nexps, xmin, xmax, npoints -> specific to exp dec
        """
        super(ExpDecSpectrum, self).__init__(hs, fs, framerate, full)
        self.y0 = y0
        self.amp = amp
        self.tau = tau
        self.sig = sig
        self.Nexps = Nexps
        self.xmin = xmin
        self.xmax = xmax
        self.npoints = npoints

    def make_exp_dec_dataframe(self, tau, sig, amp):
        # create pandas dataframe
        waveIDtau = str(round(tau, 5))
        waveIDsig = str(round(sig, 5))
        waveIDamp = str(round(amp, 5))
        waveID = ' t=' + waveIDtau + ' s=' + waveIDsig + ' a=' + waveIDamp

        datum = OrderedDict()
        datum['Frequency'+waveID] = self.fs
        datum['Complex'+waveID] = self.hs
        datum['Real'+waveID] = self.real
        datum['Imaginary'+waveID] = self.imag
        datum['Magnitude'+waveID] = self.amps*len(self.hs)
        datum['Amplitude'+waveID] = self.amps
        datum['Phase'+waveID] = self.angles
        datum['framerate'+waveID] = self.framerate
        df = pd.DataFrame(datum)
        print(df)
        return df

    def save_exp_dec_spectrum_txt(self, filename):
        '''creates txt Spectrum_+filename and puts dataframe of time and vals
        '''
        '''creates txt file only from ys and ts'''
        # create pandas dataframe
        namelist = filename.split('_')
        tau= float(namelist[2])
        sig= float(namelist[4])
        amp= float(namelist[6])
        df = self.make_exp_dec_dataframe(tau, sig, amp)
        path = r'C:\Python\_FFT_fitting_project\spectrum_data\\'
        df.to_csv(path + filename, header=True, index=False, sep='\t', mode='w')
        return

class IntegratedSpectrum:
    """Represents the integral of a spectrum."""

    def __init__(self, cs, fs):
        """Initializes an integrated spectrum:

        cs: sequence of cumulative amplitudes
        fs: sequence of frequencies
        """
        self.cs = np.asanyarray(cs)
        self.fs = np.asanyarray(fs)

    def plot_power(self, low=0, high=None, expo=False, **options):
        """Plots the integrated spectrum.

        low: int index to start at
        high: int index to end at
        """
        cs = self.cs[low:high]
        fs = self.fs[low:high]

        if expo:
            cs = np.exp(cs)

        thinkplot.plot(fs, cs, **options)

    def estimate_slope(self, low=1, high=-12000):
        """Runs linear regression on log cumulative power vs log frequency.

        returns: slope, inter, r2, p, stderr
        """
        # print self.fs[low:high]
        # print self.cs[low:high]
        x = np.log(self.fs[low:high])
        y = np.log(self.cs[low:high])
        t = scipy.stats.linregress(x, y)
        return t


class Dct(_SpectrumParent):
    """Represents the spectrum of a signal using discrete cosine transform."""

    @property
    def amps(self):
        """Returns a sequence of amplitudes (read-only property).

        Note: for DCTs, amps are positive or negative real.
        """
        return self.hs

    def __add__(self, other):
        """Adds two DCTs elementwise.

        other: DCT

        returns: new DCT
        """
        if other == 0:
            return self

        assert self.framerate == other.framerate
        hs = self.hs + other.hs
        return Dct(hs, self.fs, self.framerate)

    __radd__ = __add__

    def make_wave(self):
        """Transforms to the time domain.

        returns: Wave
        """
        N = len(self.hs)
        ys = scipy.fftpack.idct(self.hs, type=2) / 2 / N
        # NOTE: whatever the start time was, we lose it when
        # we transform back
        # ts = self.start + np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=self.framerate)


class Spectrogram:
    """Represents the spectrum of a signal."""

    def __init__(self, spec_map, seg_length):
        """Initialize the spectrogram.

        spec_map: map from float time to Spectrum
        seg_length: number of samples in each segment
        """
        self.spec_map = spec_map
        self.seg_length = seg_length

    def any_spectrum(self):
        """Returns an arbitrary spectrum from the spectrogram."""
        index = next(iter(self.spec_map))
        return self.spec_map[index]

    @property
    def time_res(self):
        """Time resolution in seconds."""
        spectrum = self.any_spectrum()
        return float(self.seg_length) / spectrum.framerate

    @property
    def freq_res(self):
        """Frequency resolution in Hz."""
        return self.any_spectrum().freq_res

    def times(self):
        """Sorted sequence of times.

        returns: sequence of float times in seconds
        """
        ts = sorted(iter(self.spec_map))
        return ts

    def frequencies(self):
        """Sequence of frequencies.

        returns: sequence of float freqencies in Hz.
        """
        fs = self.any_spectrum().fs
        return fs

    def plot(self, high=None, **options):
        """Make a pseudocolor plot.

        high: highest frequency component to plot
        """
        fs = self.frequencies()
        i = None if high is None else find_index(high, fs)
        fs = fs[:i]
        ts = self.times()

        # make the array
        size = len(fs), len(ts)
        array = np.zeros(size, dtype=np.float)

        # copy amplitude from each spectrum into a column of the array
        for j, t in enumerate(ts):
            spectrum = self.spec_map[t]
            array[:, j] = spectrum.amps[:i]

        thinkplot.pcolor(ts, fs, array, **options)

    def make_wave(self):
        """Inverts the spectrogram and returns a Wave.

        returns: Wave
        """
        res = []
        for t, spectrum in sorted(self.spec_map.items()):
            wave = spectrum.make_wave()
            n = len(wave)

            window = 1 / np.hamming(n)
            wave.window(window)

            i = wave.find_index(t)
            start = i - n // 2
            end = start + n
            res.append((start, end, wave))

        starts, ends, waves = zip(*res)
        low = min(starts)
        high = max(ends)

        ys = np.zeros(high - low, np.float)
        for start, end, wave in res:
            ys[start:end] = wave.ys

        # ts = np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=wave.framerate)


class Wave:
    """Represents a discrete-time waveform.

    """

    def __init__(self, ys, ts=None, framerate=None):
        """Initializes the wave.

        ys: wave array
        ts: array of times
        framerate: samples per second
        """
        self.ys = np.asanyarray(ys)
        self.framerate = framerate if framerate is not None else 11025

        if ts is None:
            self.ts = np.arange(len(ys)) / self.framerate
        else:
            self.ts = np.asanyarray(ts)

    def get_ts(self):
        return self.ts
    def get_ys(self):
        return self.ys

    def copy(self):
        """Makes a copy.

        Returns: new Wave
        """
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.ys)

    @property
    def start(self):
        return self.ts[0]

    @property
    def end(self):
        return self.ts[-1]

    @property
    def duration(self):
        """Duration (property).

        returns: float duration in seconds
        """
        return len(self.ys) / self.framerate

    def __add__(self, other):
        """Adds two waves elementwise.

        other: Wave

        returns: new Wave
        """
        if other == 0:
            return self

        assert self.framerate == other.framerate

        # make an array of times that covers both waves
        start = min(self.start, other.start)
        end = max(self.end, other.end)
        n = int(round((end - start) * self.framerate)) + 1
        ys = np.zeros(n)
        ts = start + np.arange(n) / self.framerate

        def add_ys(wave):
            i = find_index(wave.start, ts)

            # make sure the arrays line up reasonably well
            diff = ts[i] - wave.start
            dt = 1 / wave.framerate
            if (diff / dt) > 0.1:
                warnings.warn("Can't add these waveforms; their "
                              "time arrays don't line up.")

            j = i + len(wave)
            ys[i:j] += wave.ys

        add_ys(self)
        add_ys(other)

        return Wave(ys, ts, self.framerate)

    __radd__ = __add__

    def __or__(self, other):
        """Concatenates two waves.

        other: Wave

        returns: new Wave
        """
        if self.framerate != other.framerate:
            raise ValueError('Wave.__or__: framerates do not agree')

        ys = np.concatenate((self.ys, other.ys))
        # ts = np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=self.framerate)

    def __mul__(self, other):
        """Multiplies two waves elementwise.

        Note: this operation ignores the timestamps; the result
        has the timestamps of self.

        other: Wave

        returns: new Wave
        """
        # the spectrums have to have the same framerate and duration
        assert self.framerate == other.framerate
        assert len(self) == len(other)

        ys = self.ys * other.ys
        return Wave(ys, self.ts, self.framerate)

    def max_diff(self, other):
        """Computes the maximum absolute difference between waves.

        other: Wave

        returns: float
        """
        assert self.framerate == other.framerate
        assert len(self) == len(other)

        ys = self.ys - other.ys
        return np.max(np.abs(ys))

    def convolve(self, other):
        """Convolves two waves.

        Note: this operation ignores the timestamps; the result
        has the timestamps of self.

        other: Wave or NumPy array

        returns: Wave
        """
        if isinstance(other, Wave):
            assert self.framerate == other.framerate
            window = other.ys
        else:
            window = other

        ys = np.convolve(self.ys, window, mode='full')
        # ts = np.arange(len(ys)) / self.framerate
        return Wave(ys, framerate=self.framerate)

    def diff(self):
        """Computes the difference between successive elements.

        returns: new Wave
        """
        ys = np.diff(self.ys)
        ts = self.ts[1:].copy()
        return Wave(ys, ts, self.framerate)

    def cumsum(self):
        """Computes the cumulative sum of the elements.

        returns: new Wave
        """
        ys = np.cumsum(self.ys)
        ts = self.ts.copy()
        return Wave(ys, ts, self.framerate)

    def quantize(self, bound, dtype):
        """Maps the waveform to quanta.

        bound: maximum amplitude
        dtype: numpy data type or string

        returns: quantized signal
        """
        return quantize(self.ys, bound, dtype)

    def apodize(self, denom=20, duration=0.1):
        """Tapers the amplitude at the beginning and end of the signal.

        Tapers either the given duration of time or the given
        fraction of the total duration, whichever is less.

        denom: float fraction of the segment to taper
        duration: float duration of the taper in seconds
        """
        self.ys = apodize(self.ys, self.framerate, denom, duration)

    def hamming(self):
        """Apply a Hamming window to the wave.
        """
        self.ys *= np.hamming(len(self.ys))

    def window(self, window):
        """Apply a window to the wave.

        window: sequence of multipliers, same length as self.ys
        """
        self.ys *= window

    def scale(self, factor):
        """Multplies the wave by a factor.

        factor: scale factor
        """
        self.ys *= factor

    def shift(self, shift):
        """Shifts the wave left or right in time.

        shift: float time shift
        """
        # TODO: track down other uses of this function and check them
        self.ts += shift

    def roll(self, roll):
        """Rolls this wave by the given number of locations.
        """
        self.ys = np.roll(self.ys, roll)

    def truncate(self, n):
        """Trims this wave to the given length.

        n: integer index
        """
        self.ys = truncate(self.ys, n)
        self.ts = truncate(self.ts, n)

    def zero_pad(self, n):
        """Trims this wave to the given length.

        n: integer index
        """
        self.ys = zero_pad(self.ys, n)
        self.ts = self.start + np.arange(n) / self.framerate

    def normalize(self, amp=1.0):
        """Normalizes the signal to the given amplitude.

        amp: float amplitude
        """
        self.ys = normalize(self.ys, amp=amp)

    def unbias(self):
        """Unbiases the signal.
        """
        self.ys = unbias(self.ys)

    def find_index(self, t):
        """Find the index corresponding to a given time."""
        n = len(self)
        start = self.start
        end = self.end
        i = round((n - 1) * (t - start) / (end - start))
        return int(i)

    def segment(self, start=None, duration=None):
        """Extracts a segment.

        start: float start time in seconds
        duration: float duration in seconds

        returns: Wave
        """
        if start is None:
            start = self.ts[0]
            i = 0
        else:
            i = self.find_index(start)

        j = None if duration is None else self.find_index(start + duration)
        return self.slice(i, j)

    def slice(self, i, j):
        """Makes a slice from a Wave.

        i: first slice index
        j: second slice index
        """
        ys = self.ys[i:j].copy()
        ts = self.ts[i:j].copy()
        return Wave(ys, ts, self.framerate)

    def make_spectrum(self, full=False, normalize=False):
        """Computes the spectrum using FFT.

        returns: Spectrum
        """
        n = len(self.ys)
        d = 1 / self.framerate

        if full:
            if normalize:
                hs = np.fft.fft(self.ys)/n
            else:
                hs = np.fft.fft(self.ys)

            fs = np.fft.fftfreq(n, d)

        else:
            if normalize:
                hs = np.fft.rfft(self.ys)/n
            else:
                hs = np.fft.rfft(self.ys)
            fs = np.fft.rfftfreq(n, d)

        return Spectrum(hs, fs, self.framerate, full)

    def make_dct(self):
        """Computes the DCT of this wave.
        """
        N = len(self.ys)
        hs = scipy.fftpack.dct(self.ys, type=2)
        fs = (0.5 + np.arange(N)) / 2
        return Dct(hs, fs, self.framerate)

    def make_spectrogram(self, seg_length, win_flag=True):
        """Computes the spectrogram of the wave.

        seg_length: number of samples in each segment
        win_flag: boolean, whether to apply hamming window to each segment

        returns: Spectrogram
        """
        if win_flag:
            window = np.hamming(seg_length)
        i, j = 0, seg_length
        step = seg_length // 2

        # map from time to Spectrum
        spec_map = {}

        while j < len(self.ys):
            segment = self.slice(i, j)
            if win_flag:
                segment.window(window)

            # the nominal time for this segment is the midpoint
            t = (segment.start + segment.end) / 2
            spec_map[t] = segment.make_spectrum()

            i += step
            j += step

        return Spectrogram(spec_map, seg_length)

    def get_xfactor(self, options):
        try:
            xfactor = options['xfactor']
            options.pop('xfactor')
        except KeyError:
            xfactor = 1
        return xfactor

    def plot(self, **options):
        """Plots the wave.

        """
        xfactor = self.get_xfactor(options)
        thinkplot.plot(self.ts * xfactor, self.ys, **options)

    def plot_vlines(self, **options):
        """Plots the wave with vertical lines for samples.

        """
        xfactor = self.get_xfactor(options)
        thinkplot.vlines(self.ts * xfactor, 0, self.ys, **options)

    def corr(self, other):
        """Correlation coefficient two waves.

        other: Wave

        returns: float coefficient of correlation
        """
        corr = np.corrcoef(self.ys, other.ys)[0, 1]
        return corr

    def cov_mat(self, other):
        """Covariance matrix of two waves.

        other: Wave

        returns: 2x2 covariance matrix
        """
        return np.cov(self.ys, other.ys)

    def cov(self, other):
        """Covariance of two unbiased waves.

        other: Wave

        returns: float
        """
        total = sum(self.ys * other.ys) / len(self.ys)
        return total

    def cos_cov(self, k):
        """Covariance with a cosine signal.

        freq: freq of the cosine signal in Hz

        returns: float covariance
        """
        n = len(self.ys)
        factor = math.pi * k / n
        ys = [math.cos(factor * (i + 0.5)) for i in range(n)]
        total = 2 * sum(self.ys * ys)
        return total

    def cos_transform(self):
        """Discrete cosine transform.

        returns: list of frequency, cov pairs
        """
        n = len(self.ys)
        res = []
        for k in range(n):
            cov = self.cos_cov(k)
            res.append((k, cov))

        return res

    def write(self, filename='sound.wav'):
        """Write a wave file.

        filename: string
        """
        print('Writing', filename)
        wfile = WavFileWriter(filename, self.framerate)
        wfile.write(self)
        wfile.close()

    def play(self, filename='sound.wav'):
        """Plays a wave file.

        filename: string
        """
        self.write(filename)
        play_wave(filename)

    def make_audio(self):
        """Makes an IPython Audio object.
        """
        audio = Audio(data=self.ys.real, rate=self.framerate)
        return audio
    def get_ys(self):
        return self.ys
    def get_ts(self):
        return self.ts

    def save_wave_txt(self, filename):
        '''creates txt Signal_+tau+sig+amp and puts dataframe of time and vals'''

        '''creates txt file only from ys and ts'''
        # create pandas dataframe
        datum = OrderedDict()
        datum['time'] = self.ts
        datum['Wave_vals'] = self.ys
        df = pd.DataFrame(datum)
        # waveID = str(round(self)) # modify ?
        print('saving to file ', 'Signal_'+filename)
        df.to_csv('Signal_'+filename, header=True, index=False, sep='\t', mode='w')
        return

class ExpDecWave(Wave):
    def __init__(self, y0, amp, tau, sig, Nexps, xmin, xmax, npoints, \
                 ys, ts=None, framerate=None):
        """Initializes the ExpDecWave. Same as other waves but with additional parameters
        ys: wave array
        ts: array of times
        framerate: samples per second
        y0, amp, tau, sig, Nexps, xmin, xmax, npoints -> specific to exp dec
        """
        super(ExpDecWave, self).__init__(ys, ts, framerate)
        self.y0 = y0
        self.amp = amp
        self.tau = tau
        self.sig = sig
        self.Nexps = Nexps
        self.xmin = xmin
        self.xmax = xmax
        self.npoints = npoints

    def make_exp_dec_spectrum(self, full=False):
        """Computes the spectrum using FFT.

        returns: Spectrum
        """
        n = len(self.ys)
        d = 1 / self.framerate

        if full:
            hs = np.fft.fft(self.ys)
            fs = np.fft.fftfreq(n, d)
        else:
            hs = np.fft.rfft(self.ys)
            fs = np.fft.rfftfreq(n, d)

        return ExpDecSpectrum(self.y0, self.amp, self.tau, self.sig, \
                              self.Nexps, self.xmin, self.xmax, self.npoints, \
                 hs, fs, framerate=self.framerate, full=False)

    def save_wave_txt(self, filename=None):
        '''creates txt Signal_+tau+sig+amp and puts dataframe of time and vals
        rewrited for exp dec'''
        if filename:
            print('filename is passed for exp dec ', filename)

        '''creates txt file only from ys and ts'''
        # create pandas dataframe
        datum = OrderedDict()
        datum['time'] = self.ts
        datum['Wave_vals'] = self.ys
        df = pd.DataFrame(datum)
        print('saving to file ', filename)
        path = r'C:\Python\_FFT_fitting_project\signal_data\\'
        df.to_csv(path + filename, header=True, index=False, sep='\t', mode='w')
        return


def get_wave_from_txt(filename):
    '''takes txt Signal_+filename and puts dataframe of time and vals
    '''
    print('reading file', 'Signal_'+filename)
    df = pd.read_csv('Signal_'+filename, sep='\t')
    ts = df['time'].values
    ys = df['Wave_vals'].values
    framerate = infer_framerate(ts)
    return Wave(ts=ts, ys=ys, framerate=framerate)

def get_exp_dec_wave_from_txt(tau, sig, amp):
    '''takes txt Signal_+filename and puts dataframe of time and vals
    '''
    waveIDtau = str(round(tau, 5))
    waveIDsig = str(round(sig, 5))
    waveIDamp = str(round(amp, 5))
    waveID = '_tau_' + waveIDtau + '_sig_' + waveIDsig + '_amp_' + waveIDamp
    print('getting file ', 'Signal' + waveID + '_.txt')
    path = r'C:\Python\_FFT_fitting_project\signal_data\\'
    full_file_name = path + 'Signal' + waveID + '_.txt'
    df = pd.read_csv(full_file_name, index_col = False, sep='\t') # check without header
    # print(df)
    ts = np.array(df.time)
    ys = np.array(df.Wave_vals)
    framerate = infer_framerate(ts)
    Npoints = framerate
    start = ts[0]
    end = ts[-1]
    y0 = 0.01
    Nexps = 100
    return ExpDecWave(y0, amp, tau, sig, Nexps, xmin=start, xmax=end, npoints=Npoints,\
                      ys=ys, ts=ts, framerate=framerate)

def get_option_from_txt(tau, sig, amp, option):
    '''takes option Frequency Complex Real Imaginary Magnitude Amplitude or Phase
    '''
    ''' [0 Frequency, 1 Complex, 2 Real, 3 Imaginary, 4 Magnitude, 5 Amplitude, 6 Phase]'''
    ilocations = {"Frequency":0, 'Complex':1,'Real':2,\
               "Imaginary":3,'Magnitude':4,'Amplitude':5, 'Phase':6}
    waveIDtau = str(round(tau, 5))
    waveIDsig = str(round(sig, 5))
    waveIDamp = str(round(amp, 5))
    waveID = '_tau_' + waveIDtau + '_sig_' + waveIDsig + '_amp_' + waveIDamp
    path = r'C:\Python\_FFT_fitting_project\spectrum_data\\'
    full_file_name = path + 'Spectrum' + waveID + '_.txt'
    print('getting file ', 'Spectrum' + waveID + '_.txt')
    df = pd.read_csv(full_file_name, index_col = False, sep='\t') # check without header
    # print(df)
    xs = np.array(df.iloc[:,0]) # get freqs
    index = ilocations[option]
    ys = np.array(df.iloc[:,index])

    return xs,ys

# datum['Frequency' + waveID] = self.fs
# datum['Complex' + waveID] = self.hs
# datum['Real' + waveID] = self.real
# datum['Imaginary' + waveID] = self.imag
# datum['Magnitude' + waveID] = self.amps * len(self.hs)
# datum['Amplitude' + waveID] = self.amps
# datum['Phase' + waveID] = self.angles
# datum['framerate' + waveID] = self.framerate

def unbias(ys):
    """Shifts a wave array so it has mean 0.

    ys: wave array

    returns: wave array
    """
    return ys - ys.mean()


def normalize(ys, amp=1.0):

    """Normalizes a wave array so the maximum amplitude is +amp or -amp.
    ys: wave array
    amp: max amplitude (pos or neg) in result
    returns: wave array
    """

    high, low = abs(max(ys)), abs(min(ys))
    return amp * ys / max(high, low)


def shift_right(ys, shift):
    """Shifts a wave array to the right and zero pads.

    ys: wave array
    shift: integer shift

    returns: wave array
    """
    res = np.zeros(len(ys) + shift)
    res[shift:] = ys
    return res


def shift_left(ys, shift):
    """Shifts a wave array to the left.

    ys: wave array
    shift: integer shift

    returns: wave array
    """
    return ys[shift:]


def truncate(ys, n):
    """Trims a wave array to the given length.

    ys: wave array
    n: integer length

    returns: wave array
    """
    return ys[:n]


def quantize(ys, bound, dtype):
    """Maps the waveform to quanta.

    ys: wave array
    bound: maximum amplitude
    dtype: numpy data type of the result

    returns: quantized signal
    """
    if max(ys) > 1 or min(ys) < -1:
        warnings.warn('Warning: normalizing before quantizing.')
        ys = normalize(ys)

    zs = (ys * bound).astype(dtype)
    return zs


def apodize(ys, framerate, denom=20, duration=0.1):
    """Tapers the amplitude at the beginning and end of the signal.

    Tapers either the given duration of time or the given
    fraction of the total duration, whichever is less.

    ys: wave array
    framerate: int frames per second
    denom: float fraction of the segment to taper
    duration: float duration of the taper in seconds

    returns: wave array
    """
    # a fixed fraction of the segment
    n = len(ys)
    k1 = n // denom

    # a fixed duration of time
    k2 = int(duration * framerate)

    k = min(k1, k2)

    w1 = np.linspace(0, 1, k)
    w2 = np.ones(n - 2 * k)
    w3 = np.linspace(1, 0, k)

    window = np.concatenate((w1, w2, w3))
    return ys * window


class Signal:
    """Represents a time-varying signal.
    Remember Signal has no evaluate method"""

    def __add__(self, other):
        """Adds two signals.

        other: Signal

        returns: Signal
        """
        if other == 0:
            return self
        return SumSignal(self, other)

    __radd__ = __add__

    @property
    def period(self):
        """Period of the signal in seconds (property).

        Since this is used primarily for purposes of plotting,
        the default behavior is to return a value, 0.1 seconds,
        that is reasonable for many signals.

        returns: float seconds
        """
        return 0.1

    def plot(self, framerate=11025):
        """Plots the signal.

        The default behavior is to plot three periods.

        framerate: samples per second
        """
        duration = self.period * 3
        wave = self.make_wave(duration, start=0, framerate=framerate)
        wave.plot()

    def make_wave(self, duration=1, start=0, framerate=11025):
        """Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        n = round(duration * framerate)
        if type(duration) == float or type(framerate) == float or type(start) == float:
            n = duration * framerate
        ts = start + np.arange(n) / framerate
        ys = self.evaluate(ts)
        return Wave(ys, ts, framerate=framerate)


def infer_framerate(ts):
    """Given ts, find the framerate.

    Assumes that the ts are equally spaced.

    ts: sequence of times in seconds

    returns: frames per second
    """
    # TODO: confirm that this is never used and remove it
    dt = ts[1] - ts[0]
    framerate = 1.0 / dt
    return framerate


class SumSignal(Signal):
    """Represents the sum of signals."""

    def __init__(self, *args):
        """Initializes the sum.

        args: tuple of signals
        """
        self.signals = args

    @property
    def period(self):
        """Period of the signal in seconds.

        Note: this is not correct; it's mostly a placekeeper.

        But it is correct for a harmonic sequence where all
        component frequencies are multiples of the fundamental.

        returns: float seconds
        """
        return max(sig.period for sig in self.signals)

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        return sum(sig.evaluate(ts) for sig in self.signals)


class Sinusoid(Signal):
    """Represents a sinusoidal signal."""

    def __init__(self, freq=440, amp=1.0, offset=0, func=np.sin):
        """Initializes a sinusoidal signal.

        freq: float frequency in Hz
        amp: float amplitude, 1.0 is nominal max
        offset: float phase offset in radians
        func: function that maps phase to amplitude
        """
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.func = func

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return 1.0 / self.freq

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * self.func(phases)
        return ys


def CosSignal(freq=440, amp=1.0, offset=0):
    """Makes a cosine Sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 is nominal max
    offset: float phase offset in radians

    returns: Sinusoid object
    """
    return Sinusoid(freq, amp, offset, func=np.cos)


def SinSignal(freq=440, amp=1.0, offset=0):
    """Makes a sine Sinusoid.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 is nominal max
    offset: float phase offset in radians

    returns: Sinusoid object
    """
    return Sinusoid(freq, amp, offset, func=np.sin)


def Sinc(freq=440, amp=1.0, offset=0):
    """Makes a Sinc function.

    freq: float frequency in Hz
    amp: float amplitude, 1.0 is nominal max
    offset: float phase offset in radians

    returns: Sinusoid object
    """
    return Sinusoid(freq, amp, offset, func=np.sinc)


class ComplexSinusoid(Sinusoid):
    """Represents a complex exponential signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        phases = PI2 * self.freq * ts + self.offset
        ys = self.amp * np.exp(1j * phases)
        return ys


class SquareSignal(Sinusoid):
    """Represents a square signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        ys = self.amp * np.sign(unbias(frac))
        return ys


class SawtoothSignal(Sinusoid):
    """Represents a sawtooth signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        ys = normalize(unbias(frac), self.amp)
        return ys


class ParabolicSignal(Sinusoid):
    """Represents a parabolic signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        ys = (frac - 0.5) ** 2
        ys = normalize(unbias(ys), self.amp)
        return ys


class CubicSignal(ParabolicSignal):
    """Represents a cubic signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ys = ParabolicSignal.evaluate(self, ts)
        ys = np.cumsum(ys)
        ys = normalize(unbias(ys), self.amp)
        return ys


class GlottalSignal(Sinusoid):
    """Represents a periodic signal that resembles a glottal signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        ys = frac ** 2 * (1 - frac)
        ys = normalize(unbias(ys), self.amp)
        return ys


class TriangleSignal(Sinusoid):
    """Represents a triangle signal."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        cycles = self.freq * ts + self.offset / PI2
        frac, _ = np.modf(cycles)
        ys = np.abs(frac - 0.5)
        ys = normalize(unbias(ys), self.amp)
        return ys


class Chirp(Signal):
    """Represents a signal with variable frequency."""

    def __init__(self, start=440, end=880, amp=1.0):
        """Initializes a linear chirp.

        start: float frequency in Hz
        end: float frequency in Hz
        amp: float amplitude, 1.0 is nominal max
        """
        self.start = start
        self.end = end
        self.amp = amp

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return ValueError('Non-periodic signal.')

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        freqs = np.linspace(self.start, self.end, len(ts) - 1)
        return self._evaluate(ts, freqs)

    def _evaluate(self, ts, freqs):
        """Helper function that evaluates the signal.

        ts: float array of times
        freqs: float array of frequencies during each interval
        """
        dts = np.diff(ts)
        dps = PI2 * freqs * dts
        phases = np.cumsum(dps)
        phases = np.insert(phases, 0, 0)
        ys = self.amp * np.cos(phases)
        return ys


class ExpoChirp(Chirp):
    """Represents a signal with varying frequency."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        start, end = np.log10(self.start), np.log10(self.end)
        freqs = np.logspace(start, end, len(ts) - 1)
        return self._evaluate(ts, freqs)


class SilentSignal(Signal):
    """Represents silence."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        return np.zeros(len(ts))


class Impulses(Signal):
    """Represents silence."""

    def __init__(self, locations, amps=1):
        self.locations = np.asanyarray(locations)
        self.amps = amps

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ys = np.zeros(len(ts))
        indices = np.searchsorted(ts, self.locations)
        # this is very interresting
        ys[indices] = self.amps
        return ys


class _Noise(Signal):
    """Represents a noise signal (abstract parent class)."""

    def __init__(self, amp=1.0):
        """Initializes a white noise signal.

        amp: float amplitude, 1.0 is nominal max
        """
        self.amp = amp

    @property
    def period(self):
        """Period of the signal in seconds.

        returns: float seconds
        """
        return ValueError('Non-periodic signal.')


class UncorrelatedUniformNoise(_Noise):
    """Represents uncorrelated uniform noise."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ys = np.random.uniform(-self.amp, self.amp, len(ts))
        return ys


class UncorrelatedGaussianNoise(_Noise):
    """Represents uncorrelated gaussian noise."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ys = np.random.normal(0, self.amp, len(ts))
        return ys


class BrownianNoise(_Noise):
    """Represents Brownian noise, aka red noise."""

    def evaluate(self, ts):
        """Evaluates the signal at the given times.

        Computes Brownian noise by taking the cumulative sum of
        a uniform random series.

        ts: float array of times

        returns: float wave array
        """
        dys = np.random.uniform(-1, 1, len(ts))
        # ys = scipy.integrate.cumtrapz(dys, ts)
        ys = np.cumsum(dys)
        ys = normalize(unbias(ys), self.amp)
        return ys


class PinkNoise(_Noise):
    """Represents Brownian noise, aka red noise."""

    def __init__(self, amp=1.0, beta=1.0):
        """Initializes a pink noise signal.

        amp: float amplitude, 1.0 is nominal max
        """
        self.amp = amp
        self.beta = beta

    def make_wave(self, duration=1, start=0, framerate=11025):
        """Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        signal = UncorrelatedUniformNoise()
        wave = signal.make_wave(duration, start, framerate)
        spectrum = wave.make_spectrum()

        spectrum.pink_filter(beta=self.beta)

        wave2 = spectrum.make_wave()
        wave2.unbias()
        wave2.normalize(self.amp)
        return wave2


class ExpDecGaussTau(Signal):
    """Represents ExpDecGaussTau."""

    def __init__(self, y0, amp, tau, sig, Nexps, xmin, xmax, npoints):
        self.y0 = y0
        self.amp = amp
        self.tau = tau
        self.sig = sig
        self.Nexps = Nexps
        self.xmin = xmin
        self.xmax = xmax
        self.npoints = npoints

    def evaluate_previous(self, ts):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        ys = exp_dec_gauss_tau(x=ts, y0=self.y0, amp=self.amp, tau=self.tau, \
                            sig=self.sig, Nexps=self.Nexps, xmin=self.xmin, \
                            xmax=self.xmax, points=self.npoints)
        return ys

    def evaluate(self, ts, Npoints = 20001):
        """Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        ys = exp_dec_gauss_tau(x=ts, y0=self.y0, amp=self.amp, tau=self.tau, \
                            sig=self.sig, Nexps=self.Nexps, xmin=self.xmin, \
                            xmax=self.xmax, points=self.npoints)
        return ys

    def make_exp_dec_wave(self, y0, amp, tau, sig, Nexps, xmin, xmax, npoints,\
                          duration=1, start=0, framerate=11025):
        """Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        # working with floats
        n = round(duration * framerate)
        # ts = start + np.arange(n) / framerate
        ts = np.linspace(xmin, xmax, npoints)
        ys = self.evaluate(ts)
        return ExpDecWave(y0, amp, tau, sig, Nexps, xmin, xmax, npoints, \
                        ys, ts, framerate=framerate,)

    def save_exp_dec_signal(self):
        '''creates txt Signal_+tau+sig+amp and puts dataframe of time and vals
        '''
        '''creates txt file only from ys and ts'''
        # todo resolve ts and ys issue
        # create pandas dataframe
        datum = OrderedDict()
        # doesnt have ts and ys
        # datum['time'] = self.ts # doesnt have ts and ys
        # datum['Wave_vals'] = self.ys # doesnt have ts and ys
        df = pd.DataFrame(datum)
        waveIDtau = str(round(self.tau, 5))
        waveIDsig = str(round(self.sig, 5))
        waveIDamp = str(round(self.amp, 5))
        waveID = 'tau_'+ waveIDtau + 'sig_'+ waveIDsig+ 'amp_'+ waveIDamp
        print('saving to file ', 'Signal_' + waveID)
        path = r'C:\Python\_FFT_fitting_project\signal_data\\'
        df.to_csv(path + 'Signal'+ waveID + '_.txt', header=True, index=False, sep='\t', mode='w')
        return

def exp_dec_gauss_tau(x, y0, amp, tau, sig, Nexps, xmin, xmax, points: int):  # import
    # todo build function from npoints total
    # y0,amp,tau,sig,Nexps
    taus = list(range(Nexps))
    lgtaus = list(range(Nexps))
    amps = list(range(Nexps))
    lgstep = 1
    y1 = 0
    lgtau = m.log10(tau)
    lgsig = m.log10(sig)
    if sig < 1:
        sig = 1 / sig
    lg_tmax = lgtau + 3 * lgsig
    lg_tmin = lgtau - 3 * lgsig
    tmax = 10 ** lg_tmax
    tmin = 10 ** lg_tmin
    lgstep = (lg_tmax - lg_tmin) / Nexps  # 200.0
    # define xlist
    # xarray = np.arange()
    # define ylist
    yarray = list(range(len(x)))
    for yi in range(len(yarray)):
        y = 0
        for i in range(1, Nexps):
            lgtaus[i] = lg_tmin + i * lgstep
            taus[i] = 10 ** lgtaus[i]
            amps[i] = m.exp(-((lgtaus[i] - lgtau) ** 2) / (2 * lgsig ** 2))
            y += amps[i] * m.exp(-abs(x[yi]) / taus[i])
            pass
        y = amp * y + y0
        yarray[yi] = y

    yarray = np.array(yarray)
    # y2 = A2*exp(-(tlg-t02)*(tlg-t02)/2/w2/w2)
    return yarray

def plot_log(xs, ys): # now in exp_dec_analysis.py
    plt.figure()
    plt.plot(xs, ys)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def plot_log_with_points(xs, ys, indlist): #  now in exp_dec_analysis.py
    plt.figure(dpi=256, figsize=(20, 30))
    plt.plot(xs, ys)
    xpoints = list(xs[i] for i in indlist)
    ypoints = list(ys[i] for i in indlist)
    print(ypoints)
    plt.scatter(xpoints, ypoints)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def rest(duration):
    """Makes a rest of the given duration.

    duration: float seconds

    returns: Wave
    """
    signal = SilentSignal()
    wave = signal.make_wave(duration)
    return wave

def make_note(midi_num, duration, sig_cons=CosSignal, framerate=11025):
    """Make a MIDI note with the given duration.

    midi_num: int MIDI note number
    duration: float seconds
    sig_cons: Signal constructor function
    framerate: int frames per second

    returns: Wave
    """
    freq = midi_to_freq(midi_num)
    signal = sig_cons(freq)
    wave = signal.make_wave(duration, framerate=framerate)
    wave.apodize()
    return wave

def make_chord(midi_nums, duration, sig_cons=CosSignal, framerate=11025):
    """Make a chord with the given duration.

    midi_nums: sequence of int MIDI note numbers
    duration: float seconds
    sig_cons: Signal constructor function
    framerate: int frames per second

    returns: Wave
    """
    freqs = [midi_to_freq(num) for num in midi_nums]
    signal = sum(sig_cons(freq) for freq in freqs)
    wave = signal.make_wave(duration, framerate=framerate)
    wave.apodize()
    return wave


def midi_to_freq(midi_num):
    """Converts MIDI note number to frequency.

    midi_num: int MIDI note number

    returns: float frequency in Hz
    """
    x = (midi_num - 69) / 12.0
    freq = 440.0 * 2 ** x
    return freq


def sin_wave(freq, duration=1, offset=0):
    """Makes a sine wave with the given parameters.

    freq: float cycles per second
    duration: float seconds
    offset: float radians

    returns: Wave
    """
    signal = SinSignal(freq, offset=offset)
    wave = signal.make_wave(duration)
    return wave

def cos_wave(freq, duration=1, offset=0):
    """Makes a cosine wave with the given parameters.

    freq: float cycles per second
    duration: float seconds
    offset: float radians

    returns: Wave
    """
    signal = CosSignal(freq, offset=offset)
    wave = signal.make_wave(duration)
    return wave

def mag(a):
    """Computes the magnitude of a numpy array.

    a: numpy array

    returns: float
    """
    return np.sqrt(np.dot(a, a))


def zero_pad(array, n):
    """Extends an array with zeros.

    array: numpy array
    n: length of result

    returns: new NumPy array
    """
    res = np.zeros(n)
    res[:len(array)] = array
    return res

def create_tau_sig_amp():
    """:returns
     taulist [0.01832, 0.04979, 0.36788, 1.0, 20.08554, 54.59815, 148.41316]
     siglist [1.0001, 1.64872, 2.71828, 7.38906, 20.08554]
     amplist [0.00248, 0.01832, 0.13534, 1.0, 7.38906]"""

    def DischargeByExp(start, end):
        # find logs
        lnStart = int(m.log(start))
        # print('lnStart', lnStart)
        lnEnd = int(m.log(end))
        # print('lnEnd', lnEnd)
        l = []
        for ln in range(lnStart, lnEnd+1, 1):
            l.append(round(m.exp(ln),5))
            # print(round(m.exp(ln),2))
        return l

    addsig = round(m.e**0.5, 5) # to add this point

    taulist = DischargeByExp(0.01, 150)
    siglist = DischargeByExp(1.001, 50)
    amplist = DischargeByExp(0.001, 10)

    def leave_odd(l):
        L = []
        for i in range(0, len(l), 2):
            L.append(l[i])
        return L

    def leave_most_tau(taulist:list)->list:
        '''leave most part'''
        return taulist[0:2] + taulist[3:5] + taulist[7:9] + taulist[9:len(taulist)]
    # discharge taulist (take odd)
    # discharge amplist (take odd)
    # add sig = e**0.5
    taulist = leave_most_tau(taulist)
    siglist.insert(1, addsig)
    siglist[0] = 1.0001
    amplist = leave_odd(amplist)
    return taulist, siglist, amplist

def get_p1_p2_points(ys:array):
    # todo get fst make lg_fst
    # todo get lst make lg_lst
    # todo diff = lg_fst - lg_lst
    # todo p1 find diff*0.95
    # todo p2 find diff*0.05
    # things work like this 7.3|-(6.935)-------0-----------(0.365)--|0
    # things work like this 2.5|-(2.135)-------0----------(-4.435)--|-4.8
    # things work like this   delta1-----------0----------------delta2
    fst = ys[1]
    lg_fst = math.log10(fst)
    lst = ys[-1]
    lg_lst = math.log10(lst)
    assert lg_fst > lg_lst
    diff = lg_fst - lg_lst

    delta1 = diff - diff*0.95
    lg_p1_target = lg_fst-delta1
    p1_target = 10**(lg_p1_target)

    for ind, el in enumerate(ys):
        p1val = el
        if el <= p1_target:
            p1 = ind
            break

    delta2 = diff*0.05
    lg_p2_target = lg_lst+delta2
    p2_target = 10**(lg_p2_target)

    l = len(ys)
    for ind in range(l-1, 1, -1):
        p2val = ys[ind]
        if ys[ind] >= p2_target:
            p2 = ind
            break
    return p1, p2

def get_point_95(ys:array):
    '''returns index of point with 95% of log diff maximum'''
    fst = ys[1]
    lg_fst = math.log10(fst)
    for ind, el in enumerate(ys):
        if el <= fst*0.95:
            return ind

def get_point_05(ys:array):
    '''returns index of point with 5% of maximum'''
    fst = ys[1]
    l = len(ys)
    for ind in range(l-1, 1, -100):
        yind = ys[ind]
        if ys[ind] <= fst*0.05:
            return ind

def make_wave_and_save(tau, sig, amp): # deprecated, now in exp_dec_analysis.py
    # produce filename
    waveIDtau = str(round(tau, 5))
    waveIDsig = str(round(sig, 5))
    waveIDamp = str(round(amp, 5))
    waveID = '_tau_' + waveIDtau + '_sig_' + waveIDsig + '_amp_' + waveIDamp
    file_name = 'Signal' + waveID + '_.txt'
    # get the wave
    start = -169.5
    end = 169.5
    npoints_total = 20001
    frame_rate = npoints_total / (end - start)

    exp_dec_signal = ExpDecGaussTau(y0=0.01, tau=tau, \
                             sig=sig, amp=amp, \
                             Nexps=100, xmin=start, \
                             xmax=end, npoints=npoints_total)

    wave_from_exp_dec = exp_dec_signal.make_exp_dec_wave(y0=0.01, tau=tau, \
                                                  sig=sig, amp=amp, \
                                                  Nexps=100, xmin=start, \
                                                  xmax=end, npoints=npoints_total, \
                                                  duration=end - start, start=start, framerate=frame_rate)
    # save wave
    wave_from_exp_dec.save_wave_txt(filename=file_name)
    return wave_from_exp_dec

def make_spectrum_and_save(wave, tau, sig, amp): # deprecated, now in exp_dec_analysis.py
    # produce filename
    waveIDtau = str(round(tau, 5))
    waveIDsig = str(round(sig, 5))
    waveIDamp = str(round(amp, 5))
    waveID = '_tau_' + waveIDtau + '_sig_' + waveIDsig + '_amp_' + waveIDamp
    file_name = 'Spectrum' + waveID + '_.txt'
    # get spectrum
    spectrum_from_exp_dec = wave.make_exp_dec_spectrum()
    # spectrum_from_exp_dec.plot()
    # plt.show()

    # save it
    spectrum_from_exp_dec.save_exp_dec_spectrum_txt(filename=file_name)
    return spectrum_from_exp_dec

def running_exp_decs(mode1 = 'test', mode2 = 'load'):
    taulist, siglist, amplist = create_tau_sig_amp()
    print('running')
    if mode1 == 'test':
        """
         taulist [0.01832, 0.04979, 0.36788, 1.0, 20.08554, 54.59815, 148.41316]
         siglist [1.0001, 1.64872, 2.71828, 7.38906, 20.08554]
         amplist [0.00248, 0.01832, 0.13534, 1.0, 7.38906]"""
        tau = taulist[3]
        sig = siglist[4]
        amp = amplist[0]
        if mode2 == 'save':
            wave = make_wave_and_save(tau, sig, amp)
            make_spectrum_and_save(wave, tau, sig, amp)
        elif mode2 == 'load':
            wave = get_exp_dec_wave_from_txt(tau, sig, amp)  # tau, sig, amp should be strictly from the list
            spec = wave.make_spectrum()
            # print(spec)
            spec.smart_plot()
            plt.show()

            # xs, ys = get_option_from_txt(tau, sig, amp, 'Amplitude')
            # p1, p2 = get_p1_p2_points(ys)
            # plot_log_with_points(xs, ys, (p1, p2))

            # plot_log(xs,ys)

        # wave.plot()
        # plt.show()

    elif mode1 == 'run':
        for tau in taulist[:4]:
            for sig in siglist[:1]:
                for amp in amplist[:1]:
                    if mode2 == 'save':
                        wave = make_wave_and_save(tau, sig, amp)
                        spectrum = wave.make_exp_dec_spectrum()
                        make_spectrum_and_save(wave, tau, sig, amp)
                    elif mode2 == 'load':
                        # wave = get_exp_dec_wave_from_txt(tau, sig, amp)
                        print(tau)
                        xs, ys = get_option_from_txt(tau, sig, amp, 'Amplitude')
                        # point95 = get_point_95(ys)
                        # point05 = get_point_05(ys)
                        # plot_log_with_points(xs, ys, [point95, point05])
                        for y in ys[:50]:
                            print(y)
                        for y in ys[:-50]:
                            print(y)
                        plot_log(xs,ys)

        return

if __name__ == '__main__':
    print('starting')
    # running_exp_decs(mode1 = 'test', mode2 = 'load')
