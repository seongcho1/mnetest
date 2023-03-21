import numpy as np
import os
import mne

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

from ft_utils import raw_filenames, fetch_data, prepare_data, filter_data, fetch_events



def ft_fit(SUBJECTS, RUNS, tmin=-0.2, tmax=0.5, forceplot=False):

    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames(SUBJECTS, RUNS), runs=RUNS)))
    labels, epochs = fetch_events(raw, tmin=tmin, tmax=tmax)

    # spectrum = raw.compute_psd()
    # p = spectrum.plot_topomap()

    #epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    epochs_train = epochs.copy().crop(tmin=-0.2, tmax=0.5)
    #epochs_train = epochs.copy().crop(tmin=tmin, tmax=tmax)

    epochs_data = epochs.get_data()
    print(f'epochs_data.shape={epochs_data.shape}')
    epochs_data_train = epochs_train.get_data()
    print(f'epochs_data_train.shape={epochs_data_train.shape}')

    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(1, test_size=0.2, random_state=42)
    #cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
    csp = FT_CSP(n_components=4, reg=None, log=True, norm_trace=False)
    #csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    # print('X=epochs_data_train, y=labels')
    # csp.fit_transform(epochs_data_train, labels)
    # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    # print('X=epochs_data, y=labels')
    # csp.fit_transform(epochs_data, labels)
    # csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

    # return

    print('# Use scikit-learn Pipeline with cross_val_score function')
    clf = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
    # scores_ldashrinkage = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    # mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)

    ###############################
    epochs_data_train_csp = csp.fit_transform(epochs_data_train, labels)
    clf2 = Pipeline([('LDA', lda_shrinkage)])
    scores_ldashrinkage = cross_val_score(clf2, epochs_data_train_csp, labels, cv=cv, n_jobs=1)
    mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)
    ###############################

    print('# Printing the results')
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print('-'*42)
    print("LDA SHRINKED Classification accuracy: %f / Chance level: %f" % (np.mean(scores_ldashrinkage), class_balance))
    print(f"Mean Score Model {mean_scores_ldashrinkage}")
    print(f"Std Score Model {std_scores_ldashrinkage}")
    print('-'*42)

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)
    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

    # save pipeline
    clf = clf.fit(epochs_data_train, labels)
    dump(clf, "final_model.joblib")
    print("model saved to final_model.joblib")

    return raw


if __name__ == "__main__":

    #%matplotlib qt5

    #DATA_DIR = "mne_data"

    RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
    RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    RUNS = RUNS2

    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)

    plt.ioff()
    SUBJECTS = [13]

    subject_list=[ 1,  3,  4,  8,  9, 12, 13, 15, 17, 18,
                  19, 20, 21, 22, 25, 26, 36, 37, 40, 41,
                  42, 46, 47, 48, 50, 51, 53, 54, 61, 62,
                  63, 68, 71, 73, 77, 80, 83, 84, 85, 86,
                  87, 90, 93, 98, 100, 101, 102, 103, 104, 105]

    raw = ft_fit(SUBJECTS, RUNS, tmin=tmin, tmax=tmax)

    # plt.ion()
    # fig = plt.figure(figsize=(4.2, 4.2))
    # plt.plot(range(10), range(10))
    plt.show()


    # PREDICT_MODEL = "final_model.joblib"
    # SUBJECTS = [2]
    # ft_predict()

    # ft_pipeline()
