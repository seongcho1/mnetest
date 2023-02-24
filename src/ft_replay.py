import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from collections import deque

from ft_utils import raw_filenames, fetch_data, prepare_data, filter_data, fetch_events


#https://towardsdatascience.com/dynamic-replay-of-time-series-data-819e27212b4b

def prt(x):
    print(repr(x))

def get_eeg():
    RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
    RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    RUNS = RUNS2

    SUBJECTS = [1]


    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames(SUBJECTS, RUNS), runs=RUNS)))
    print("raw", type(raw), repr(raw))

    # spectrum = raw.compute_psd()
    # spectrum.plot_topomap()

    arr = np.asarray(raw[:][0])
    arr_t = arr.transpose()

    chs = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..']
    eeg = pd.DataFrame(arr_t, columns=chs)
    print('eeg', eeg.shape)
    print(eeg.head())

    print('-+'*42)
    eeg = eeg * 1000000
    print(eeg.head())

    return raw, eeg

def replay(eeg):
    #%matplotlib qt5
    #matplotlib.use('qtagg')

    ##################################
    sfreq = 500 # sampling frequency
    visible = 2000 # time shown in plot (in samples) --> 4 seconds

    # initialize deques
    dy1 = deque(np.zeros(visible), visible)
    dy2 = deque(np.zeros(visible), visible)
    dx = deque(np.zeros(visible), visible)

    # get interval of entire time frame
    interval = np.linspace(0, eeg.shape[0], num=eeg.shape[0])
    interval /= sfreq # from samples to seconds

    # define channels to plot
    ch1 = 'Fp2.'
    ch2 = 'C3..'


    ##################################
    # define figure size
    fig = plt.figure(figsize=(12,12))

    # define axis1, labels, and legend
    ah1 = fig.add_subplot(211)
    ah1.set_ylabel("Voltage [\u03BCV]", fontsize=14)
    l1, = ah1.plot(dx, dy1, color='rosybrown', label=ch1)
    ah1.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)

    # define axis2, labels, and legend
    ah2 = fig.add_subplot(212)
    ah2.set_xlabel("Time [s]", fontsize=14, labelpad=10)
    ah2.set_ylabel("Voltage [\u03BCV]", fontsize=14)
    l2, = ah2.plot(dx, dy2, color='silver', label=ch2)
    ah2.legend(loc="upper right", fontsize=12, fancybox=True, framealpha=0.5)
    ##################################

    start = 0

    # simulate entire data
    while start+visible <= eeg.shape[0]:

        # extend deques (both x and y axes)
        dy1.extend(eeg[ch1].iloc[start:start+visible])
        dy2.extend(eeg[ch2].iloc[start:start+visible])
        dx.extend(interval[start:start+visible])

        # update plot
        l1.set_ydata(dy1)
        l2.set_ydata(dy2)
        l1.set_xdata(dx)
        l2.set_xdata(dx)

        # get mean of deques
        mdy1 = np.mean(dy1)
        mdy2 = np.mean(dy2)

        # set x- and y-limits based on their mean
        ah1.set_ylim(-120+mdy1, 200+mdy1)
        ah1.set_xlim(interval[start], interval[start+visible])
        ah2.set_ylim(-60+mdy2, 100+mdy2)
        ah2.set_xlim(interval[start], interval[start+visible])

        # control speed of moving time-series
        start += 25

        fig.canvas.draw()
        fig.canvas.flush_events()

    ##################################
    ##################################
    ##################################
    ##################################

    # print("head", arr2[:][0:5])
    # labels, epochs = fetch_events(raw)
    # print(epochs.get_data().shape)
    # (45, 64, 801)


if __name__ == "__main__":
    plt.ioff()
    #plt.ion()
    raw, eeg = get_eeg()
    plt.show()

    spectrum = raw.compute_psd()
    p = spectrum.plot_topomap()


    plt.ion()
    replay(eeg)


