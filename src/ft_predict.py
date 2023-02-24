import time
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

from ft_utils import raw_filenames, fetch_data, prepare_data, filter_data, fetch_events
from ft_fit import ft_fit

def ft_predict(SUBJECTS, RUNS):
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    start = time.perf_counter()

    # Fetch Data
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames(SUBJECTS, RUNS), runs=RUNS)))
    labels, epochs = fetch_events(raw)
    # labels, epochs = fetch_events(filter_data(raw))
    epochs = epochs.get_data()

    print("X shape= ", epochs.shape, "y shape= ", labels.shape)

    scores = []
    for n in range(epochs.shape[0]):
        pred = clf.predict(epochs[n:n + 1, :, :])
        print("pred= ", pred, "truth= ", labels[n:n + 1])
        scores.append(1 - np.abs(pred[0] - labels[n:n + 1][0]))
    print("Mean acc= ", np.mean(scores))


    end = time.perf_counter()
    exectime = end - start

    unit = 's'
    if exectime < 0.001:
        exectime *= 1000
        unit = 'ms'

    print(f"===(exec-time = {exectime:.3f} {unit})===")


if __name__ == "__main__":
    #DATA_DIR = "mne_data"

    RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
    RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    RUNS = RUNS2

    SUBJECTS = [6]

    ft_fit(SUBJECTS, RUNS)

    PREDICT_MODEL = "final_model.joblib"
    #SUBJECTS = [41]
    ft_predict(SUBJECTS, RUNS)

    # ft_pipeline()

    # 1,   2,     3,    4,     5,   6,     42
    #-----------------------------------------------------------------
    # 0.8, 0.867, 0.73, 0.667, 0.8, 0.8    0.867   #with filter(7, 30)
    #  avg of [1, 2, 3, 4] = 0.766
    #-----------------------------------------------------------------
    # 0.82 0.756  0.711 0.644  0.822 0.844 0.867   #with filter(8, 40)
    #  avg of [1, 2, 3, 4] = 0.732
