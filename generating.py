# todo sig -> HWHM
import thinkdsp as dsp
import thinkplot
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from thinkdsp import Signal, Wave
import numpy as np
from collections import OrderedDict
import os
import scipy.signal as sig
import math as m
from tf_exp_funcs_29_05 import save_for_origin, save_results_np

np.linspace(0.6, 0.9, 4)

# np.insert(a, obj=)

# np.where(a<3)[0]
# sig1 = 0.5
# tau1 = 10
# tau2 = 100
# sig1 * tau1 / tau2

                                   # l = ["G_Tau_Dec T = 100", "G_Tau_Dec T = 120"]
# f_name = ' & '.join(l)
# full_name = 'G_Tau_Dec ' + ''.join(f_name.split('G_Tau_Dec '))
# full_name
# full_name = "G_Tau_Dec A=5 T=4 & G_Tau_Dec A=1 T=10 & G_Tau_Dec A=2 T =35 Brown A=1"
# # full_name = "S_exp A = 17 T = 58 G_Tau_Dec A=1 T=10 & G_Tau_Dec A=2 T =35 Brown A=1"
# # full_name = "G_Tau_Dec A=5 T=4 Brown A=1"
# if full_name.startswith('G_Tau_Dec'):
#     final = 'G_Tau_Dec '+''.join(full_name.split('G_Tau_Dec '))
# else:
#     final = full_name
# final

PI2 = np.pi*2

# If name is too long keep only types of signals
NAME_THRESHOLD = 150
SHORT_NAME_DICT = {'ExpDecGaussTau': 'G_Tau_Dec'}

def multiple_replace(text, wordDict):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text

def find_nearest_idx(array, value):
    ''' Find nearest value is an array '''
    idx = (np.abs(array - value)).argmin()
    return idx

"""all evaluate functions: Evaluate the signal at the given times.
ts: float array of times
returns: float wave array
"""

def exp_dec_single(c:float, amp:float, tau:float):
    '''c: bias, amp: amplitude, tau: decay time constant'''
    def _exp_dec_single(xs):
        return c + amp * np.exp(-xs / tau)
    return _exp_dec_single


def exp_dec_gauss_ln_tau_external(amp, tau_val, sig, n_exps, min_tau, max_tau, c=0, to_plot=False):

    assert min_tau <= tau_val <= max_tau

    def exp_dec_gauss_ln_tau_internal(xs:np.ndarray)->np.ndarray:
        ''':returns ys: np.array
        amps expected tobe all positive or all negative'''
        # ts, ys
        xs == xs.ravel()
        ys = np.zeros(shape=(xs.size,))
        # constants
        # taus = np.linspace(min_tau, max_tau, n_exps)
        taus = np.logspace(np.log(min_tau), np.log(max_tau), n_exps, base=np.e)

        lntaus = np.log(taus)
        lntau_val = np.log(tau_val)

        # sum of amps must be equal to amp
        amps = 1/(np.sqrt(PI2*np.power(sig,2))) * np.exp(-1 / 2 * ((lntaus - lntau_val) / sig) ** 2)
        amps /= amps.sum()
        amps *= amp

        # summ_ = np.sum(exponent)
        # a = amp/summ_
        # amps = a*exponent

        # fill ys if pos => only > 1.0e-15 counted, if neg => only < -1.0e-15 counted
        if amps[0] >= 0:
            nonzero_inds = np.where(amps > 1.0e-15)[0]
        else:
            nonzero_inds = np.where(amps < -1.0e-15)[0]

        for tau_i in nonzero_inds:
            ys += amps[tau_i] * np.exp(-xs / taus[tau_i])
        ys += c
        if to_plot:
            plt.figure('tau distribution')
            plt.scatter(taus, amps, s=1)
            plt.xlabel('sum(amps)'+str(sum(amps)))
            plt.ylim([amps.min(), amps.max()])
            plt.xscale('log')
            plt.show()
        return ys, amps, taus

    return exp_dec_gauss_ln_tau_internal


def step_distr_external(amp, tau_val, sig, n_exps, min_tau, max_tau, c=0, to_plot=False):

    assert min_tau <= tau_val <= max_tau

    def step_distr_internal(xs:np.ndarray)->np.ndarray:
        ''':returns ys: np.array'''
        # ts, ys
        xs == xs.ravel()
        ys = np.zeros(shape=(xs.size,))
        # constants
        # taus = np.linspace(min_tau, max_tau, n_exps)
        taus = np.logspace(np.log(min_tau), np.log(max_tau), n_exps, base=np.e)
        lntaus = np.log(taus)
        lntau_val = np.log(tau_val)

        step = np.zeros(shape=taus.shape)
        step_inds = np.where(lntau_val + sig/2 > lntaus)[0]
        step[step_inds] = 1
        non_step_inds = np.where(lntau_val - sig / 2 > lntaus)[0]
        step[non_step_inds] = 0

        amps = step
        amps /= amps.sum()
        amps *= amp

        nonzero_inds = np.where(step > 1.0e-15)[0]
        # fill ys
        for tau_i in nonzero_inds:
            ys += amps[tau_i] * np.exp(-xs / taus[tau_i])
        ys += c
        if to_plot:
            plt.figure('tau distribution')
            plt.scatter(taus, amps, s=1)
            plt.xlabel('sum(amps)'+str(sum(amps)))
            plt.ylim([amps.min(), amps.max()])
            plt.xscale('log')
            plt.show()
        return ys, amps, taus

    return step_distr_internal


def test_distr(amp, tau_val, sig, n_exps, min_tau, max_tau, c=0):
    xs = np.linspace(1, 100,100)
    internal = exp_dec_gauss_ln_tau_external(amp, tau_val, sig, n_exps, min_tau, max_tau)
    ys1, amps1, taus1 = internal(xs)
    assert np.sum(amps1) == amp
    print('np.sum(amps1)', np.sum(amps1))

    step_internal = step_distr_external(amp, tau_val, sig, n_exps, min_tau, max_tau)
    ys2, amps2, taus2 = step_internal(xs)
    assert np.sum(amps2) == amp
    print('np.sum(amps2)', np.sum(amps2))

    print('test completed')
    return

# test_distr(amp=15, tau_val=10, sig=0.3, n_exps=100, min_tau=1, max_tau=100, c=0)

def meander_function(t:np.array, t_width, T_period, w, limit):
    y = []
    for ti in t:
        summ = 0
        for k in range(1, limit+1):
            summ+=(1/k*m.sin((k*w*t_width)/2)*m.cos(k*w*ti))
        y.append(t_width/T_period + 2/m.pi * summ)
    return np.array(y)

# meander = MeanderSignal(t_width=10, T_period=10, w=0.1, limit=10000)
# meander_wave = meander.make_wave(start = -10, duration = 20, framerate = 100)
# meander_wave.plot()

class WaveExtended(Wave):
    '''
    Extended version of Wave from ThinkDSP
    '''

    def __init__(self, ys, ts=None, framerate=None, info='no info'):

        ''' gets all what Wave get plus info about all the generated Signals included
        :param ys:
        :param ts:
        :param framerate:
        :param info:
        to_repr: representation letter for Origin
        nsignals: number of signals included in Wave
        '''
        # todo mb set up to_repr and nsignals in init phase

        Wave.__init__(self, ys, ts=ts, framerate=framerate)
        assert type(info) == str
        self.info = info
        self.to_repr = 'T'
        self.nsignals = 1

        # if ts is None:
        #     self.ts = np.arange(len(ys)) / self.framerate
        # else:
        #     self.ts = np.asanyarray(ts)
        #
        # self.framerate = framerate if framerate is not None else 11025

    def __repr__(self):
        return 'Wave_info={}'.format(Wave.info)

    def save_txt(self, path, name=None, to_differentiate=True):
        '''
        Saves Wave with convenient representation in OriginLab software
        '''

        if name is not None: filename = name
        else: filename = self.info+'_.txt'

        if len(filename) <= NAME_THRESHOLD:
            multiple_replace(text=filename, wordDict=SHORT_NAME_DICT)

        save_for_origin(self.ts, self.ys, n_signals=self.nsignals, to_repr=self.to_repr,  data_path=path, name=filename, to_differentiate = to_differentiate)
        # np.savetxt(fname=fullname, X=X.T, header='ts, ys', delimiter='\t')


class SignalExtended(Signal):
    '''
    Extended version of Signal from ThinkDSP
    '''

    def make_wave(self, duration=1, start=0, framerate=11025):
        """
        Makes a Wave_extended object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        stop = duration-start

        n = round(duration * framerate)
        # ts = start + np.arange(n) / framerate
        ts = np.logspace(np.log(start), np.log(stop), n, base=np.e)
        ys = self.evaluate(ts)
        return WaveExtended(ys, ts, framerate=framerate, info=self.__repr__())

    def __add__(self, other):
        """
        Adds two signals.
        other: Signal
        returns: SumSignal
        """
        if other == 0:
            return self
        return SumSignalExtended(self, other)


class SumSignalExtended(SignalExtended):
    """
    Represents the sum of signals.
    """

    def __init__(self, *args):

        """Initializes the sum.
        args: tuple of signals"""

        self.signals = args

    def evaluate(self, ts):

        """
        Evaluates sum of the signals at the given times.
        ts: float array of times
        returns: float wave array
        """

        ts = np.asarray(ts)
        ys = np.zeros(shape=ts.shape)
        for sig in self.signals:
            ys += sig.evaluate(ts)
        return ys
        # return sum(sig.evaluate(ts) for sig in self.signals)

    def __repr__(self):

        """
        SumSignal with convenient representation in OriginLab software.
        Unites the names of all included signals
        """

        full_name = '_&_'.join((signal.__repr__() for signal in self.signals))
        if full_name.startswith('G_Tau_Dec'):
            d1 = full_name.split('G_Tau_Dec_')
            d = 'G_Tau_Dec_'+''.join(full_name.split('G_Tau_Dec')[1:])
            return 'G_Tau_Dec_'+''.join(full_name.split('G_Tau_Dec_'))
        else:
            return full_name

    def __add__(self, other):
        """
        Adds sum of signals.
        other: Signal or SumSignal
        returns: SumSignal
        """

        if other == 0:
            return self
        return SumSignalExtended(self, other)

class ExpDecSingleTauSignal(SignalExtended):
    """
    Represents Exponential decay with single time constant.
    """

    def __init__(self, c, amp, tau):
        '''
        :param c: constant to be added
        :param amp: amplitude of signal
        :param tau: time constant
        '''

        self.c = c
        self.amp = amp
        self.tau = tau

    def evaluate(self, ts):
            ts = np.asarray(ts)
            exp_dec_single_int = exp_dec_single(self.c, self.amp, self.tau)
            ys = exp_dec_single_int(ts)
            return ys

    def __repr__(self):
        return 'S_Tau_C={},_A={},_T={}'.format(
        round(self.c, 7) ,
        round(self.amp, 7) ,
        round(self.tau, 5))

class ExpDecGaussTauSignal(SignalExtended):
    """
    Represents exponents distributed by Gaussian function with mean tau.
    """

    def __init__(self, c, amp, tau, sig, to_plot=False, to_save_distr = False):
        self.c = c
        self.amp = amp
        self.tau = tau
        self.sig = sig
        self.taus = None
        self.amps = None
        self.to_plot = to_plot
        self.to_save_distr = to_save_distr

    def get_distribution_taus_amps(self, ts):
        ts = np.asarray(ts)
        exp_dec_gauss_tau_int = exp_dec_gauss_ln_tau_external(c=self.c, amp=self.amp, tau_val=self.tau, sig=self.sig, n_exps=ts.size*100 , min_tau=ts[0], max_tau=ts[-1], to_plot=False) # todo ts.size*100
        ys, self.amps, self.taus = exp_dec_gauss_tau_int(ts)
        return self.taus, self.amps

    def evaluate(self, ts):
            ts = np.asarray(ts)
            exp_dec_gauss_tau_int = exp_dec_gauss_ln_tau_external(c=self.c, amp=self.amp, tau_val=self.tau, sig=self.sig, n_exps=ts.size*100, min_tau=ts[0], max_tau=ts[-1], to_plot=False)
            ys, self.amps, self.taus = exp_dec_gauss_tau_int(ts) # todo ts.size*100
            if self.to_save_distr:
                save_distribution(self.taus, self.amps, path_to_save = '.', signal_name= self.__repr__())
            return ys


    def __repr__(self):
        # return 'G_Tau_Dec_T={}, S={}'.format(
        # round(self.tau, 3),
        # round(self.sig, 3)
        # )

        return 'G_Tau_Dec C={}, A={}, T={}, S={}'.format(
        round(self.c, 7),
        round(self.amp, 7) ,
        round(self.tau, 3) ,
        round(self.sig, 3) )

class StepTauSignal(SignalExtended):
    """
    Represents exponents distributed by Step function with mean tau.
    """

    def __init__(self, c, amp, tau, sig, to_plot=False, to_save_distr = False):
        self.c = c
        self.amp = amp
        self.tau = tau
        self.sig = sig
        self.taus = None
        self.amps = None
        self.to_plot = to_plot
        self.to_save_distr = to_save_distr

    def get_distribution_taus_amps(self, ts):
        ts = np.asarray(ts)
        step_tau_int = step_distr_external(c=self.c, amp=self.amp, tau_val=self.tau, sig=self.sig, n_exps=ts.size*100 , min_tau=ts[0], max_tau=ts[-1], to_plot=False) # todo ts.size*100
        ys, self.amps, self.taus = step_tau_int(ts)
        return self.taus, self.amps

    def evaluate(self, ts):
            ts = np.asarray(ts)
            step_tau_int = step_distr_external(c=self.c, amp=self.amp, tau_val=self.tau, sig=self.sig, n_exps=ts.size*100, min_tau=ts[0], max_tau=ts[-1], to_plot=False)
            ys, self.amps, self.taus = step_tau_int(ts) # todo ts.size*100
            if self.to_save_distr:
                save_distribution(self.taus, self.amps, path_to_save = '.', signal_name= self.__repr__(), to_repr='S')
            return ys


    def __repr__(self):
        # return 'Step_Tau_T={}, S={}'.format(
        # round(self.tau, 3),
        # round(self.sig, 3)
        # )

        return 'Step_Tau_T C={}, A={}, T={}, S={}'.format(
        round(self.c, 7) ,
        round(self.amp, 7) ,
        round(self.tau, 3) ,
        round(self.sig, 3) )

def save_distribution(taus, amps, path_to_save = '.',  signal_name = 'No_name_signal', nsignals=1, to_repr = 'T'):
    '''
    Saving distribution with convenient representation in OriginLab software
    :param taus: array of time constants
    :param amps: array of amplitudes
    :param path_to_save: path to save the distribution
    :param signal_name: full name
    :param nsignals: nsignals used in full distribution
    :param to_repr: letter to represent
    :return:
    '''
    save_for_origin(taus, amps, to_repr=to_repr, data_path=path_to_save, name='Distr_{}.txt'.format(signal_name), to_differentiate = False, x_name='taus', y_name = 'amps', n_signals=nsignals)

class MeanderSignal(SignalExtended):
    """
    Represents a square signal - meander
    """
    def __init__(self, t_width, T_period, w, limit):
        self.t_width = t_width
        self.T_period = T_period
        self.w = w
        self.limit = limit

    def evaluate(self, ts):
        """
        Evaluates the signal at the given times.

        ts: float array of times

        returns: float wave array
        """
        ts = np.asarray(ts)
        ys = meander_function(t=ts, t_width=self.t_width, T_period=self.T_period, w=self.w, limit=self.limit)
        return ys

    def make_wave(self, duration=1, start=0, framerate=11025):
        """
        Makes a Wave object.

        duration: float seconds
        start: float seconds
        framerate: int frames per second

        returns: Wave
        """
        n = int(duration*framerate)
        ts = np.linspace(start, duration+start, n)
        ys = self.evaluate(ts)
        return WaveExtended(ys, ts, framerate=framerate)

    def __repr__(self):
        return 'MeanderSignal_width={},period={},w={},limit={}'.format(
        self.t_width ,
        self.T_period ,
        self.w ,
        self.limit)

class WhiteNoise(dsp.UncorrelatedUniformNoise, SignalExtended):
    def __repr__(self):
        return 'WN={}'.format(round(self.amp, 9))

class BrownNoise(dsp.BrownianNoise, SignalExtended):
    def __repr__(self):
        return 'BN={}'.format(round(self.amp, 9))

class GaussSignalDistribution(object):
    '''
        Represents distribution object of Gaussian function
        used for saving distributions of amplitudes in signals
    '''

    def __init__(self, gaus_signals, wave_ts, to_repr='T'):
        self.gaus_signals = gaus_signals
        self.wave_ts = wave_ts
        self.nsignals = len(gaus_signals)
        self.to_repr = to_repr

    def save_full_distribution(self, path_to_save = '.'):
        # assert all taus equal
        fst_iter = True
        signal_names = []
        for each_signal in self.gaus_signals:
            each_signal.get_distribution_taus_amps(ts=self.wave_ts)
            if fst_iter:
                taus = each_signal.taus  # taus of fst
                amps = np.zeros(shape=each_signal.amps.shape)  # shape of fst todo
                fst_iter = False
            assert all((each_signal.taus[i] == taus[i] for i in range(taus.size)))
            amps += each_signal.amps
            signal_names.append(each_signal.__repr__())

        joined_name = '_&_'.join(signal_names)
        full_name = 'G_Tau_Dec_' + ''.join(joined_name.split('G_Tau_Dec_'))

        save_distribution(taus, amps, path_to_save=path_to_save, signal_name=full_name, to_repr=self.to_repr, nsignals=self.nsignals)
        return taus, amps


def shift_by(wave, phase):
    return wave.ts - phase

def smooth_wave(wave, M):
    assert wave.ts.size > M
    gaussian_window = sig.gaussian(M=M, std=M/8)
    new_gaussian_ys = gaussian_window
    new_gaussian_ys /= sum(gaussian_window)
    new_ys = np.convolve(wave.ys, new_gaussian_ys, mode='same')
    smooth_wave = WaveExtended(new_ys, wave.ts, framerate=wave.framerate, info='sm {}'.format(wave.info))
    return smooth_wave

def plot_exp(wave):
    wave.plot()
    thinkplot.config(xlabel='ts',
                     ylabel='a',
                     xscale='log')
    plt.show()

def plot_spec(spectrum):
    spectrum.plot()
    thinkplot.config(xlabel='fs',
                     ylabel='as')
    plt.show()


def sigma_tau_comp(taus=[(10, 100), (20, 90), (30, 80)], sigs=np.linspace(0.5, 2.5, 5)):
    for tau_schedule in taus:
        print('tau_schedule', tau_schedule)
        for sig in sigs:
            print('sig = ', sig)
            exp_signals = []
            # prev_tau = tau_schedule[0]
            for each_tau in tau_schedule:
                exp_signals.append(ExpDecGaussTauSignal(c=0, amp=3.0e-2, tau=each_tau, sig=sig, to_plot=False, to_save_distr=False))
                # prev_tau = each_tau

            combined_signal = sum(exp_signals) # + white_noise + brown_noise
            combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
            combined_wave.nsignals = 2
            # combined_wave.plot()
            combined_wave.save_txt(path='.')

            # todo with distribution but small size
            final_distr = GaussSignalDistribution(gaus_signals=exp_signals, wave_ts=combined_wave.ts, to_repr='T')
            final_distr.save_full_distribution()


def two_single_tau_comp(tau_distances=np.linspace(1, 8), mid_tau=5, tau_amps=[1e-6, 1e-6], const = 1e-5):

    for distance in tau_distances:
        tau1 = mid_tau - distance/2
        tau2 = mid_tau + distance/2
        exp_signal1 = ExpDecSingleTauSignal(c=const, amp=tau_amps[0], tau=tau1)
        exp_signal2 = ExpDecSingleTauSignal(c=const, amp=tau_amps[1], tau=tau2)
        # print('1')
        # white_noise = WhiteNoise(amp=1.0e-9)
        # brown_noise = BrownNoise(amp=1.0e-9)
        # combined_signal = exp_signal0 + exp_signal1 + white_noise + brown_noise
        combined_signal = exp_signal1 + exp_signal2
        combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
        combined_wave.nsignals = 2
        combined_wave.to_repr = 'T'
        # print('3')
        combined_wave.plot()
        combined_wave.save_txt(path='.', name=combined_wave.info+'.txt', to_differentiate=True)


def two_single_tau_comp_log(tau_distances=np.linspace(1, 1), mid_tau=1, tau_amps=[1, 1], const = 0):

    for distance in tau_distances:
        logtau1 = mid_tau - distance/2
        logtau2 = mid_tau + distance/2
        exp_signal1 = ExpDecSingleTauSignal(c=const, amp=tau_amps[0], tau=np.exp(logtau1))
        exp_signal2 = ExpDecSingleTauSignal(c=const, amp=tau_amps[1], tau=np.exp(logtau2))
        # print('1')
        # white_noise = WhiteNoise(amp=1.0e-9)
        # brown_noise = BrownNoise(amp=1.0e-9)
        # combined_signal = exp_signal0 + exp_signal1 + white_noise + brown_noise
        combined_signal = exp_signal1 + exp_signal2
        combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
        combined_wave.nsignals = 2
        combined_wave.to_repr = 'T'
        # print('3')
        combined_wave.plot()
        combined_wave.save_txt(path='.', name=combined_wave.info+'.txt', to_differentiate=True)


def two_gauss_tau_comp(taus=[0.1, 1], tau_amps=[3.0e-2, 3.0e-2], sig=1.0e-5, const = 1e-5):

    assert len(taus) == len(tau_amps)


    exp_signal1 = ExpDecGaussTauSignal(c=const, amp=tau_amps[0], tau=taus[0], sig=sig, to_plot=False, to_save_distr=False)
    exp_signal2 = ExpDecGaussTauSignal(c=const, amp=tau_amps[1], tau=taus[1], sig=sig, to_plot=False, to_save_distr=False)

    combined_signal = exp_signal1 + exp_signal2
    combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
    combined_wave.nsignals = 2
    combined_wave.to_repr = 'T'


    combined_wave.save_txt(path='.', name=combined_wave.info+'.txt', to_differentiate=True)

    final_distr = GaussSignalDistribution(gaus_signals=[exp_signal1, exp_signal2], wave_ts=combined_wave.ts, to_repr='T')
    final_distr.save_full_distribution()



def gauss_tau_sigma_comp(tau=100, sigs=np.linspace(0.1, 0.35, 6)):
    for sig in sigs:
        print('sig = ', sig)
        exp_signal = ExpDecGaussTauSignal(c=1.0e-6, amp=1.0e-6, tau=tau, sig=sig, to_plot=False, to_save_distr=True)
        # white_noise = WhiteNoise(amp=1.0e-9)
        # brown_noise = BrownNoise(amp=1.0e-9)
        # combined_signal = exp_signal0 + exp_signal1 + exp_signal2 + white_noise + brown_noise
        combined_signal = exp_signal # + white_noise + brown_noise
        combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
        combined_wave.to_repr = 'S'
        combined_wave.nsignals = 1
        # combined_wave.plot()
        combined_wave.save_txt(path='.')

        final_distr = GaussSignalDistribution(gaus_signals=[exp_signal], wave_ts=combined_wave.ts, to_repr='S')
        final_distr.save_full_distribution()


def step_tau_sigma_comp(tau=50, sigs=np.linspace(0.5, 0.95, 6)):
    for sig in sigs:
        print('sig = ', sig)
        exp_signal = StepTauSignal(c=0, amp=1, tau=tau, sig=sig, to_plot=False, to_save_distr=True)
        # white_noise = WhiteNoise(amp=1.0e-9)
        # brown_noise = BrownNoise(amp=1.0e-9)
        # combined_signal = exp_signal0 + exp_signal1 + exp_signal2 + white_noise + brown_noise
        combined_signal = exp_signal # + white_noise + brown_noise
        combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
        combined_wave.to_repr = 'S'
        combined_wave.nsignals = 1
        combined_wave.save_txt(path='.')

        # final_distr = GaussSignalDistribution(gaus_signals=[exp_signal], wave_ts=combined_wave.ts, to_repr='S')
        # final_distr.save_full_distribution()


def noise_comp(tau=100, sig=0.2, br_noise_amps = np.linspace(1.0e-9, 1.0e-7, 5)):
    for br_amp in br_noise_amps:
        print('br_amp = ', br_amp)
        exp_signal = ExpDecGaussTauSignal(c=1.0e-7, amp=-1.0e-7, tau=tau, sig=sig, to_plot=False, to_save_distr=False)
        white_noise = WhiteNoise(amp=1.0e-9)
        brown_noise = BrownNoise(amp=br_amp)
        # combined_signal = exp_signal0 + exp_signal1 + exp_signal2 + white_noise + brown_noise
        combined_signal = exp_signal + white_noise + brown_noise
        combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
        combined_wave.plot()
        combined_wave.save_txt(path='.')

        final_distr = GaussSignalDistribution(gaus_signals=[exp_signal], wave_ts=combined_wave.ts, to_repr='T')
        final_distr.save_full_distribution()

def tau_amps_comp(taus=[(10, 100), (20, 90), (30, 80)], amps=np.linspace(0.5, 2.5, 5)):
    sig = 0.5
    for tau_schedule in taus:
        print('tau_schedule', tau_schedule)
        for amp in amps:
            print('sig = ', sig)
            exp_signals = []
            prev_tau = tau_schedule[0]
            for each_tau in tau_schedule:
                sig = sig*prev_tau/each_tau
                exp_signals.append(ExpDecGaussTauSignal(c=0, amp=1.0e-7, tau=each_tau, sig=sig, to_plot=False, to_save_distr=False))
                prev_tau = each_tau

            # white_noise = WhiteNoise(amp=1.0e-9)
            # brown_noise = BrownNoise(amp=1.0e-9)

            combined_signal = sum(exp_signals) # + white_noise + brown_noise
            combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
            # print('3')
            combined_wave.plot()
            combined_wave.save_txt(path='.')

            # todo with distribution but small size
            final_distr = GaussSignalDistribution(gaus_signals=exp_signals, wave_ts=combined_wave.ts)
            final_distr.save_full_distribution()

def one_single_tau_comp(taus=np.logspace(1, 600), tau_amp=1e-6, const = 1e-5):
    for tau in taus:
        exp_signal1 = ExpDecSingleTauSignal(c=const, amp=tau_amp, tau=tau)
        combined_signal = exp_signal1
        combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
        # print('3')
        combined_wave.plot()
        combined_wave.save_txt(path='.')


import numpy as np
np.linspace(0.1, 0.9, 17)

np.round(np.logspace(-1, 2, 5), 5)

np.round(np.logspace(np.log(0.001), np.log(1000), 45000, base=np.e), 5)

# np.logspace(0.5, 2.5, 9)

if __name__ == '__main__':
    start = 0.001
    stop = 1000
    npoints = 50000
    framerate = npoints//(stop-start)

    two_single_tau_comp_log(tau_distances=np.linspace(1.2, 1.4, 2), mid_tau=0, tau_amps=[1, 1], const=0)
    # two_single_tau_comp(tau_distances=np.linspace(1, 8), mid_tau=1, tau_amps=[1e-6, 1e-6], const=1e-5)
    # one_single_tau_comp(taus=np.round(np.logspace(np.log(0.0005), np.log(5000), 5, base=np.e), 5), tau_amp=1, const=0)

    # dists = np.round(np.logspace(-1, 2, 45000), 7)
    # two_gauss_tau_comp(taus=[0.1, 10], tau_amps=[3.0e-2, 3.0e-2], sig=1.0e-7, const=1e-5)
    # sigma_tau_comp(taus=[(0.1, 10), (0.2, 5), (0.5, 2), (0.7, 1.41),(0.8, 1.25)], sigs=np.linspace(1.0e-5, 2.5, 1))

    # taus = np.logspace(np.log(0.0005), np.log(5000), npoints, base=np.e)
    # amps = np.zeros(taus.shape)
    #
    # Ts = np.round(np.logspace(np.log(0.0005), np.log(5000), 5, base=np.e), 5)
    # for T in Ts:
    #     amps[find_nearest_idx(array=taus, value=T)] = 3.0e-2
    #
    # save_for_origin(taus, amps, n_signals=5, to_repr='T', data_path='.', name='distr_of_{}.txt'.format(Ts), to_differentiate=False,
    #                 x_name='taus', y_name='ini_Ts={}'.format(Ts))

    # i1 = find_nearest_idx(array=taus, value=T1)
    # i2 = find_nearest_idx(array=taus, value=T2)
    # amps[i1] = 3.0e-2
    # amps[i2] = 3.0e-2
    #
    #

    #
    # save_results_np(taus, amps, data_path='.', name='distr_T1={},T2={}.txt'.format(T1,T2), delimiter='\t')

    # taus = [0.01, 100]
    # tau_amps = [1, 1]
    # # sig=1.0e-5
    # const = 1e-5
    #
    # exp_signal1 = ExpDecSingleTauSignal(c=const, amp=tau_amps[0], tau=taus[0])
    # exp_signal2 = ExpDecSingleTauSignal(c=const, amp=tau_amps[1], tau=taus[1])
    #
    # combined_signal = exp_signal1 + exp_signal2
    # combined_wave = combined_signal.make_wave(duration=stop - start, start=start, framerate=framerate)
    # combined_wave.nsignals = 2
    # combined_wave.to_repr = 'T'
    #
    # combined_wave.save_txt(path='.', name=combined_wave.info + '.txt', to_differentiate=True)

    # final_distr = GaussSignalDistribution(gaus_signals=[exp_signal1, exp_signal2], wave_ts=combined_wave.ts, to_repr='T')
    # final_distr.save_full_distribution()

    # tau1 = 0.8
    # tau2 = 1.25
    # tau_amps = [1,1]
    # const = 1
    # exp_signal1 = ExpDecSingleTauSignal(c=const, amp=tau_amps[0], tau=tau1)
    # exp_signal2 = ExpDecSingleTauSignal(c=const, amp=tau_amps[1], tau=tau2)
    #
    # combined_signal = exp_signal1 + exp_signal2
    # combined_wave = combined_signal.make_wave(duration=stop - start, start=start, framerate=framerate)
    # combined_wave.nsignals = 2
    # combined_wave.to_repr = 'T'
    # # print('3')
    # combined_wave.save_txt(path='.', name=combined_wave.info + '.txt', to_differentiate=True)


    # two_single_tau_comp(tau_distances=np.linspace(100, 150, 2), mid_tau=100, tau_amps=[1, 1], const=1)

    # step_tau_sigma_comp(tau=1, sigs=np.round(np.logspace(np.log(0.1), np.log(3), 1, base=np.e), 2))

    # step_tau = StepTauSignal(c=0, amp=1, tau=50, sig=0.5)
    # step_tau_wave = step_tau.make_wave(duration=stop-start, start=start, framerate=framerate)
    # step_tau_wave.save_txt()

    # todo save distribution properly
    # gauss_tau_sigma_comp(tau=50, sigs=np.linspace(0.1, 0.9, 1))

    # two_single_tau_comp(tau_distances=np.linspace(50, 90, 1), mid_tau=50, tau_amps=[1e-6, 1e-6], const = 1e-5)

    # one_single_tau_comp(taus=np.logspace(np.log(1), np.log(100), 2, base=np.e), tau_amp=1e-6, const=1e-5)
    # ====================================================================================
    # # # todo noisy and mixed signal
    # gauss_signal = ExpDecGaussTauSignal(c=2.0e-7, amp=5.0e-2, tau=1, sig=0.5, to_plot=False,to_save_distr=False)
    # # exp_signal = ExpDecSingleTauSignal(c=0, amp=1, tau=5)
    # white_noise = WhiteNoise(amp=1.0e-9)
    # brown_noise = BrownNoise(amp=1.0e-8)
    # # combined_signal = gauss_signal + exp_signal
    # # combined_wave = combined_signal.make_wave(start=start, duration=stop - start, framerate=framerate)  # by points
    # # combined_wave.save_txt(path='.')
    # # combined_signal_noisy = gauss_signal + white_noise + brown_noise # + exp_signal
    # combined_signal_noisy = gauss_signal + white_noise + brown_noise
    # combined_wave = combined_signal_noisy.make_wave(start=start, duration=stop - start, framerate=framerate)
    # combined_wave.save_txt(path='.')

    # final_distr = GaussSignalDistribution(gaus_signals=[exp_signal0, exp_signal1], wave_ts=combined_wave.ts)
    # final_distr.save_full_distribution(path_to_save='.')

    # noise_wave = white_noise.make_wave(start=start, duration=stop - start, framerate=framerate)
    # exp_signal_noisy_wave = exp_signal_noisy.make_wave(start=start, duration=stop - start, framerate=framerate)
    # ====================================================================================


    # ====================================================================================
    # # todo smoothing effect
    # gaussian_window = sig.gaussian(M=100, std=15)
    # gaussian_wave = dsp.zero_pad(gaussian_window, exp_signal_noisy_wave.ys.size)
    # # thinkplot.plot(gaussian_wave)
    # gaussian_window /= gaussian_window.sum()
    # smoothed_exp_signal_noisy_ys = np.convolve(exp_signal_noisy_wave.ys, gaussian_window, mode='same')
    # smoothed_exp_signal_noisy_wave = WaveExtended(ys=smoothed_exp_signal_noisy_ys, ts=exp_signal_noisy_wave.ts, framerate=exp_signal_noisy_wave.framerate)
    # plt.plot(exp_signal_noisy_wave.ts, exp_signal_noisy_wave.ys, c='r')
    # plt.plot(smoothed_exp_signal_noisy_wave.ts, smoothed_exp_signal_noisy_wave.ys)
    # plt.show()
    # ====================================================================================


    # ====================================================================================
    # todo 3 gauss tau
    # sig1 = 0.5
    # tau1 = 40
    # tau2 = 80
    #
    # exp_signal0 = ExpDecGaussTauSignal(c=0, amp=1.0e-7, tau=0.01, sig=0.001, to_plot=False, to_save_distr=False) # 0.07 -> 0.4
    # # # print('0')
    # exp_signal1 = ExpDecGaussTauSignal(c=0, amp=1.0e-7, tau=0.1, sig=0.01, to_plot=False,to_save_distr=False)
    # # # print('1')
    # exp_signal2 = ExpDecGaussTauSignal(c=0, amp=1.0e-7, tau=2, sig=0.07, to_plot=False)
    # # # print('2')
    # white_noise = WhiteNoise(amp=1.0e-9)
    # brown_noise = BrownNoise(amp=1.0e-8)
    # # #
    # combined_signal = exp_signal0 + exp_signal1 + exp_signal2 + white_noise + brown_noise
    # combined_wave = combined_signal.make_wave(duration=stop-start, start=start, framerate=framerate)
    # # # print('3')
    # combined_wave.plot()
    # combined_wave.save_txt(path='.')
    #
    # final_distr = GaussSignalDistribution(gaus_signals=[exp_signal0, exp_signal1], wave_ts=combined_wave.ts)
    # final_distr.save_full_distribution(path_to_save='.')

    # ====================================================================================
    # todo brown check
    # brown_noise = BrownNoise(amp=1.0e-7)
    # brown_wave = brown_noise.make_wave(duration=stop-start, start=start, framerate=framerate)
    # brown_wave.plot()
    # brown_wave.save_txt(path='.')

    # ====================================================================================
    # todo simple exps
    # exp_signal0 = ExpDecSingleTauSignal(c=0, amp=-1.0e-7, tau=50)
    # # exp_signal1 = ExpDecSingleTauSignal(c=1.0e-6, amp=-1.0e-7, tau=70) # 1.0e-6
    # # # exp_signal2 = ExpDecSingleTauSignal(c=0, amp=1.0e-7, tau=0.01)
    # # print('2')
    # #
    # combined_signal_singles = exp_signal0 # + exp_signal1 # + exp_signal2 + white_noise + brown_noise
    # # print('3')
    # combined_singles_wave = combined_signal_singles.make_wave(duration=stop-start, start=start, framerate=framerate)
    # combined_singles_wave.plot()
    # combined_singles_wave.save_txt(path='.')

    # final_distr = GaussSignalDistribution(gaus_signals=[exp_signal0, exp_signal1], wave_ts=combined_singles_wave.ts)
    # final_distr.save_full_distribution(path_to_save='.')
    # ====================================================================================
    # check_distrib([0.15, 0.2, 0.25, 0.3, 0.35])
    # check_tau(taus=np.linspace(1, 200, 10))
    # sigma_tau_comp(taus=[(0.05, 3), (0.1, 1)], sigs=np.linspace(0.5, 2.8, 4))
    # sigma_tau_comp(taus=[(20, 90), (30, 80)], sigs=np.linspace(0.1, 0.8, 4))
    # noise_comp(tau=100, sig=0.2, br_noise_amps=np.linspace(1.0e-9, 1.0e-7, 5))