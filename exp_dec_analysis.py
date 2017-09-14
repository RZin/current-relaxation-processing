
from __future__ import print_function, division

from win32api import GetSystemMetrics
screen = (GetSystemMetrics[0], GetSystemMetrics[1])

from collections import OrderedDict
import array
import copy
import math

import numpy as np
import random
import scipy
import scipy.stats
import scipy.fftpack
import struct
import subprocess
import thinkplot
import warnings
import pandas as pd

from fractions import gcd
from wave import open as open_wave

import matplotlib.pyplot as plt

try:
    from IPython.display import Audio
except:
    warnings.warn("Can't import Audio from IPython.display; "
                  "Wave.make_audio() will not work.")

PI2 = math.pi * 2

from Full_digital_signal_processing import _SpectrumParent, Spectrum, Wave, Signal,\
    ExpDecSpectrum, ExpDecWave, ExpDecGaussTau, get_wave_from_txt, \
get_exp_dec_wave_from_txt, get_option_from_txt

# plot_log, plot_log_with_points,
#create_tau_sig_amp, get_p1_p2_points, make_wave_and_save, make_spectrum_and_save

def exp_dec_gauss_tau(x, y0, amp, tau, sig, Nexps, xmin, xmax, points: int):  # import
    # todo build function from npoints total
    import numpy as np
    import math as m
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

def plot_log(xs, ys):
    import pylab as plt
    plt.figure()
    plt.plot(xs, ys)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def plot_log_with_points(xs, ys, indlist,tau,sig,amp):
    plt.figure(dpi=256, figsize=screen)
    plt.plot(xs, ys)
    xpoints = list(xs[i] for i in indlist)
    ypoints = list(ys[i] for i in indlist)
    plt.scatter(xpoints, ypoints)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

def create_tau_sig_amp():
    import math as m
    """:returns
     taulist [0.01832, 0.04979, 0.36788, 1.0, 20.08554, 54.59815, 148.41316]
     siglist [1.0001, 1.64872, 2.71828, 7.38906, 20.08554]
     amplist [0.00248, 0.01832, 0.13534, 1.0, 7.38906]"""

    def DischargeByExp(start, end):
        import math as m
        # find logs
        lnStart = int(m.log(start))
        # print('lnStart', lnStart)
        lnEnd = int(m.log(end))
        # print('lnEnd', lnEnd)
        l = []
        for ln in range(lnStart, lnEnd+1, 1):
            # ap exp()
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

def get_p1_p2_points(ys:array, by_lg10=False):
    # todo get fst make lg_fst
    # todo get lst make lg_lst
    # todo diff = lg_fst - lg_lst
    # todo p1 find diff*0.95
    # todo p2 find diff*0.05
    # things work like this 7.3|-(6.935)-------0-----------(0.365)--|0
    # things work like this 2.5|-(2.135)-------0----------(-4.435)--|-4.8
    # things work like this   delta1-----------0----------------delta2
    fst = ys[1]
    if by_lg10:
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
    else:
        for ind, el in enumerate(ys):
            p1val = el
            if el <= fst*0.95:
                p1 = ind
                break

        l = len(ys)
        for ind in range(l-1, 1, -1):
            p2val = ys[ind]
            if ys[ind] >= fst*0.05:
                p2 = ind
                break
        return p1, p2

def get_point_95(ys:array):
    '''returns index of point with 95% of log diff maximum'''
    fst = ys[1]
    lg_fst = math.log10(fst)
    for ind, el in enumerate(ys):
        if el <= fst*0.85:
            return ind

def get_point_05(ys:array):
    '''returns index of point with 5% of maximum'''
    fst = ys[1]
    l = len(ys)
    for ind in range(l-1, 1, -100):
        yind = ys[ind]
        if ys[ind] <= fst*0.05:
            return ind

def get_filename(prefix, tau, sig, amp, suffix):
    waveIDtau = str(round(tau, 5))
    waveIDsig = str(round(sig, 5))
    waveIDamp = str(round(amp, 5))
    waveID = '_tau_' + waveIDtau + '_sig_' + waveIDsig + '_amp_' + waveIDamp
    file_name = prefix + waveID + suffix
    assert type(file_name) == str
    return file_name

def make_wave_and_save(tau, sig, amp):
    # get filename
    file_name = get_filename('Signal', tau, sig, amp,'_.txt')
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

def make_spectrum_and_save(wave, tau, sig, amp):
    # get filename
    file_name = get_filename('Spectrum', tau, sig, amp, '_.txt')
    # get spectrum
    spectrum_from_exp_dec = wave.make_exp_dec_spectrum()
    # spectrum_from_exp_dec.plot()
    # plt.show()
    # save it
    spectrum_from_exp_dec.save_exp_dec_spectrum_txt(filename=file_name)
    return spectrum_from_exp_dec

def plot_exp_dec_wave(wave, tau, sig, amp, to_save):
    # get filename
    file_name = get_filename('SignalGraph', tau, sig, amp, '_.png')
    xs = wave.ts
    ys = wave.ys
    plt.figure(figsize=screen)
    plt.plot(xs, ys)
    plt.title(file_name)
    plt.xlabel('t')
    plt.ylabel('y')
    if to_save:
        plt.savefig(file_name, bbox_inches='tight')
    else:
        plt.show()

def plot_exp_dec_spectrum(spectrum, tau, sig, amp, point_ind_list=None, to_save = False):
    # get filename
    path = r'C:\Python\_FFT_fitting_project\spectrum_graphs\\'
    file_name = get_filename('SpectrumGraph', tau, sig, amp, '_.png')
    xs = spectrum.fs
    ys = spectrum.amps
    plt.figure(figsize=screen)
    plt.plot(xs, ys)
    plt.xscale('log')
    # plt.yscale('log')
    plt.title(file_name)
    plt.xlabel('w')
    plt.ylabel('y')
    if point_ind_list is not None:
        xpoints = list(xs[i] for i in point_ind_list)
        ypoints = list(ys[i] for i in point_ind_list)
        plt.scatter(xpoints, ypoints)
    if to_save:
        plt.savefig(file_name)
    else:
        plt.show()

def get_exp_dec_wave(tau, sig, amp):
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
    return wave_from_exp_dec

def analyse_exp_dec(tau, sig, amp, mode2='check'):
    if mode2 == 'save':
        wave = make_wave_and_save(tau, sig, amp)
        spec = make_spectrum_and_save(wave, tau, sig, amp)
        plot_exp_dec_spectrum(spec, tau, sig, amp, point_ind_list=(0, -1), to_save=True)
    elif mode2 == 'check':
        wave = get_exp_dec_wave(tau, sig, amp)
        spec = wave.make_spectrum()
        p1, p2 = get_p1_p2_points(spec.amps)
        freqs = spec.fs
        amps = spec.amps
        print(freqs[p1], freqs[p2])
        print(amps[p1], amps[p2])
        plot_exp_dec_spectrum(spec, tau, sig, amp, point_ind_list=(p1, p2), to_save=False)
        # plt.show()
    elif mode2 == 'load':
        wave = get_exp_dec_wave_from_txt(tau, sig, amp)  # tau, sig, amp should be strictly from the list
        # plot_wave_and_save(wave, tau, sig, amp)
        spec = wave.make_spectrum()
        # get high low points
        p1, p2 = get_p1_p2_points(spec.amps, by_lg10=False)
        # plot and save
        plot_exp_dec_spectrum(spec, tau, sig, amp, point_ind_list=(p1, p2), to_save=False)
        # xs, ys = get_option_from_txt(tau, sig, amp, 'Amplitude')
        # p1, p2 = get_p1_p2_points(ys)
        # plot_log_with_points(xs, ys, (p1, p2))

def running_exp_decs(mode1 = 'test', mode2 = 'load'):
    taulist, siglist, amplist = create_tau_sig_amp()
    print('running')
    if mode1 == 'test':
        """
         taulist [0.01832, 0.04979, 0.36788, 1.0, 20.08554, 54.59815, 148.41316]
         siglist [1.0001, 1.64872, 2.71828, 7.38906, 20.08554]
         amplist [0.00248, 0.01832, 0.13534, 1.0, 7.38906]"""
        import pylab as plt
        tau = taulist[2]
        sig = siglist[0]
        amp = amplist[0]
        analyse_exp_dec(tau, sig, amp, mode2=mode2)
    elif mode1 == 'run':
        for tau in taulist[:4]:
            for sig in siglist[:1]:
                for amp in amplist[:1]:
                    analyse_exp_dec(tau, sig, amp, mode2=mode2)
        return

if __name__ == '__main__':
    print('starting')
    running_exp_decs(mode1 = 'test', mode2 = 'check')