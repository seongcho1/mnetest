import numpy as np
import os
import mne
import matplotlib.pyplot as plt

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne import events_from_annotations
from mne.channels import make_standard_montage

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ft_csp import FT_CSP  # use my own CSP
from mne.decoding import CSP  # use mne CSP

from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from joblib import dump, load

import copy as cp
from sklearn.base import BaseEstimator, TransformerMixin

def fetch_events(data_filtered, tmin=-1., tmax=4.):
    print("\n" + ">"*42*2)
    print(">>>fetch_events(data_filtered, tmin=-1., tmax=4.)<<<")
    print("<"*42*2 + "\n")

    event_ids = dict(T1=0, T2=1)
    events, _ = events_from_annotations(data_filtered, event_id=event_ids)
    picks = mne.pick_types(data_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                        picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1]
    return labels, epochs


def filter_data(raw, montage=make_standard_montage('standard_1020')):
    print("\n" + ">"*42*2)
    print(">>>filter_data(raw, montage=make_standard_montage('standard_1020'))<<<")
    print("<"*42*2 + "\n")

    data_filter = raw.copy()
    #data_filter.set_montage(montage)

    data_filter.filter(7, 30, fir_design='firwin', skip_by_annotation='edge')
    #data_filter.filter(8, 40, fir_design='firwin', skip_by_annotation='edge')

    p = mne.viz.plot_raw(data_filter, scalings={"eeg": 75e-6})
    return data_filter


def prepare_data(raw, montage=make_standard_montage('standard_1020')):
    print("\n" + ">"*42*2)
    print(">>>prepare_data(raw, montage=make_standard_montage('standard_1020'))<<<")
    print("<"*42*2 + "\n")

    raw.rename_channels(lambda x: x.strip('.'))
    eegbci.standardize(raw)
    # https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)
    raw.set_montage(montage)

    # plot
    montage = raw.get_montage()
    p = montage.plot()
    p = mne.viz.plot_raw(raw, scalings={"eeg": 75e-6})
    return raw


def fetch_data(raw_fnames, runs, sfreq=None, bPrint=True):
    print("\n" + ">"*42*2)
    print(">>>fetch_data(raw_fnames, sfreq=None, bPrint=True)<<<")
    print("<"*42*2 + "\n")

    dataset = []
    subject = []
    for i, f in enumerate(raw_fnames):
        if f.endswith(".edf") and int(f.split('R')[1].split(".")[0]) in runs:
            #subject_data = read_raw_edf(os.path.join(f"{DATA_DIR}/{SUBJECTS[0]}", f), preload=True)
            subject_data = read_raw_edf(f, preload=True)
            if sfreq is None:
                sfreq = subject_data.info["sfreq"]
            if subject_data.info["sfreq"] == sfreq:
                subject.append(subject_data)
            else:
                break

    dataset.append(mne.concatenate_raws(subject))
    raw = concatenate_raws(dataset)

    if bPrint:
        print(raw)
        print(raw.info)
        print(raw.info["ch_names"])
        print(raw.annotations)
    return raw

def raw_filenames(subjects, runs):
    print("\n" + ">"*42*2)
    print(">>>raw_filenames()<<<")
    print("<"*42*2 + "\n")

    raw_fnames = []
    for subject in subjects:
      subject_raw_fnames = eegbci.load_data(subject, runs)
      raw_fnames.extend(subject_raw_fnames)
    return raw_fnames
