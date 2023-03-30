import time
import numpy as np
import os
import mne
from random import seed ,randint

import matplotlib
matplotlib.use('qtagg')
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
from sklearn.metrics import make_scorer

from ft_utils import raw_filenames, fetch_data, prepare_data, \
                     filter_data, fetch_events, \
                     my_custom_loss_func
from ft_fit import ft_fit


def ft_predict(SUBJECTS, RUNS, tmin=-0.2, tmax=0.5, PREDICT_MODEL="final_model.joblib"):
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    start = time.perf_counter()

    # Fetch Data
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames(SUBJECTS, RUNS), runs=RUNS)))
    labels, epochs = fetch_events(raw, tmin=tmin, tmax=tmax)
    # labels, epochs = fetch_events(filter_data(raw))
    epochs = epochs.get_data()

    print(f"X.shape= {epochs.shape}, y.shape={labels.shape}")

    score = make_scorer(my_custom_loss_func, greater_is_better=False)

    scores = []
    predicts = []
    for n in range(epochs.shape[0]):
        pred = clf.predict(epochs[n:n + 1, :, :])
        print(f"event={n:02d}, predict={pred}, label={labels[n:n + 1]}")
        scores.append(pred[0] == labels[n:n + 1][0])
        predicts.append(pred[0])

    end = time.perf_counter()
    exectime = end - start

    unit = 's'
    if exectime < 0.001:
        exectime *= 1000
        unit = 'ms'

    print('='*42)

    print(f"=     (clf.predict Mean-Accuracy={np.mean(scores):.3f} )     =")
    print(f"=     (clf.predict Mean-Accuracy={score(clf, epochs, labels):.3f} )     =")
    print(f"=     (clf.predict Exec-Time    ={exectime:.3f}{unit})     =")
    print('='*42)

    return predicts, np.mean(scores)


if __name__ == "__main__":
    #DATA_DIR = "mne_data"

    RUNS1 = [3, 7, 11]  # motor: left hand vs right hand
    RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    RUNS3 = [5, 9, 13]  # motor: hands vs feet
    RUNS4 = [6, 10, 14] # motor imagery: hands vs feet
    RUNS = RUNS2
    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)


    #$SUBJECTS = [3]
    SUBJECTS = []  # [1, 2, 3, 4] # [7,8,9,10,11,12, 42]

    # seed(4)
    # for i in range(4):
    # seed(1)
    for i in range(6):
        x = randint(1, 109)
        SUBJECTS.append(x)

    #SUBJECTS = [2, 51, 78] #bad

    # SUBJECTS = [46, 80]  # [1, 2, 3, 4] # [7,8,9,10,11,12, 42]
    # SUBJECTS = [1, 3, 5, 46, 80]  # [1, 2, 3, 4] # [7,8,9,10,11,12, 42]

    # #SUBJECTS = [i + 1 for i in range(100, 109)]

    # SUBJECTS=[ 1,  3,  4,  8,  9, 12, 13, 15, 17, 18, 19, 20,
    #           21, 22, 25, 26, 36, 37, 40,
    #           41, 42, 46, 47, 48, 50, 51, 53, 54,
    #           61, 62, 63, 68, 71, 73, 77, 80,
    #           83, 84, 85, 86, 87, 90, 93, 98, 100,
    #           101, 102, 103, 104, 105]

    # SUBJECTS=[9, 15, 17, 19, #13,
    #           #21, 40,
    #           #48, 53, #41, 42, 46, 47, 50, 51,
    #           ]

    # idx = []
    # for i in range(6):
    #     x = randint(0, 49)
    #     idx.append(x)
    # SUBJECTS = [SUBJECTS1[id] for id in idx]

    scores_ = []
    predict_ = None

    #ft_fit(SUBJECTS, RUNS, tmin=tmin, tmax=tmax)
    # SUBJECTS = [i + 1 for i in range(0, 20)]
    for SUBJECT in SUBJECTS:
        #RUNS = RUNS2
        ft_fit([SUBJECT], RUNS, tmin=tmin, tmax=tmax)
        PREDICT_MODEL = "final_model.joblib"
        #SUBJECTS = [41]
        #RUNS = RUNS2
        predict_, score_ = ft_predict([SUBJECT], RUNS, tmin=tmin, tmax=tmax, PREDICT_MODEL=PREDICT_MODEL)
        scores_.append(round(score_, 2))
        print("42 predict\n", predict_)

    print("\n\nsubjects:", SUBJECTS)
    print("score   :", scores_)

    # scores_.remove(max(scores_))
    # scores_.remove(min(scores_))

    print("mean score:", round(np.mean(scores_), 2))



    # ft_fit(SUBJECTS, RUNS)

    # PREDICT_MODEL = "final_model.joblib"
    # #SUBJECTS = [41]
    # ft_predict(SUBJECTS, RUNS)

    # ft_pipeline()

    # 1,     2,     3,     4,     5,     6,     42
    #-----------------------------------------------------------------
    # 0.8,   0.867, 0.73,  0.667, 0.8,   0.8    0.867   #with filter(7, 30)
    #  avg of [1, 2, 3, 4] = 0.766
    #-----------------------------------------------------------------
    # 0.733, 0.822, 0.689  0.689, 0.867, 0.778  0.889   #with filter(7, 30), csp(log=False)
    #  avg of [1, 2, 3, 4] =

    #-----------------------------------------------------------------
    # 0.82   0.756  0.711  0.644  0.822   0.844 0.867   #with filter(8, 40)
    #  avg of [1, 2, 3, 4] = 0.732
