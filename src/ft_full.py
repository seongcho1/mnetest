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


def fetch_data(raw_fnames, sfreq=None, bPrint=True):
    print("\n" + ">"*42*2)
    print(">>>fetch_data(raw_fnames, sfreq=None, bPrint=True)<<<")
    print("<"*42*2 + "\n")

    dataset = []
    subject = []
    for i, f in enumerate(raw_fnames):
        if f.endswith(".edf") and int(f.split('R')[1].split(".")[0]) in RUNS:
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

def raw_filenames():
    print("\n" + ">"*42*2)
    print(">>>raw_filenames()<<<")
    print("<"*42*2 + "\n")

    raw_fnames = []
    for subject in SUBJECTS:
      subject_raw_fnames = eegbci.load_data(subject, RUNS)
      raw_fnames.extend(subject_raw_fnames)
    return raw_fnames



def ft_fit():

    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames())))
    labels, epochs = fetch_events(raw)

    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)

    epochs_data = epochs.get_data()
    print(f'epochs_data.shape={epochs_data.shape}')
    epochs_data_train = epochs_train.get_data()
    print(f'epochs_data_train.shape={epochs_data_train.shape}')

    # Define a monte-carlo cross-validation generator (reduce variance):
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
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

    scores_ldashrinkage = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)

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


def ft_predict():
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    # Fetch Data
    raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames())))
    labels, epochs = fetch_events(filter_data(raw))
    epochs = epochs.get_data()

    print("X shape= ", epochs.shape, "y shape= ", labels.shape)

    scores = []
    for n in range(epochs.shape[0]):
        pred = clf.predict(epochs[n:n + 1, :, :])
        print("pred= ", pred, "truth= ", labels[n:n + 1])
        scores.append(1 - np.abs(pred[0] - labels[n:n + 1][0]))
    print("Mean acc= ", np.mean(scores))


def ft_pipeline():
    #raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames())))
    #labels, epochs = fetch_events(filter_data(raw))

    raw_fnames = raw_filenames()
    raw = fetch_data(raw_fnames)
    raw = prepare_data(raw)
    raw = filter_data(raw)
    labels, epochs = fetch_events(raw)

    epochs_data_train = epochs.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    # Assemble a classifier
    lda = LDA()
    lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
    svc = SVC(gamma='auto')

    #csp = CSP()
    #csp = CSP(n_components=4, reg=None, log=None, norm_trace=False)
    csp = FT_CSP(n_components=4, reg=None, log=None, norm_trace=False)


    #classifiers
    clf1 = Pipeline([('CSP', csp), ('LDA', lda)])
    scores_lda = cross_val_score(clf1, epochs_data_train, labels, cv=cv, n_jobs=1)
    mean_scores_lda, std_scores_lda = np.mean(scores_lda), np.std(scores_lda)

    clf2 = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
    scores_ldashrinkage = cross_val_score(clf2, epochs_data_train, labels, cv=cv, n_jobs=1)
    mean_scores_ldashrinkage, std_scores_ldashrinkage = np.mean(scores_ldashrinkage), np.std(scores_ldashrinkage)

    clf3 = Pipeline([('CSP', csp), ('SVC', svc)])
    scores_svc = cross_val_score(clf3, epochs_data_train, labels, cv=cv, n_jobs=1)
    mean_scores_svc, std_scores_svc = np.mean(scores_svc), np.std(scores_svc)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)

    print('-'*42)
    print("LDA Classification accuracy: %f / Chance level: %f" % (np.mean(scores_lda), class_balance))
    print(f"Mean Score Model {mean_scores_lda}")
    print(f"Std Score Model {std_scores_lda}")
    print('-'*42)
    print("LDA SHRINKED Classification accuracy: %f / Chance level: %f" % (np.mean(scores_ldashrinkage), class_balance))
    print(f"Mean Score Model {mean_scores_ldashrinkage}")
    print(f"Std Score Model {std_scores_ldashrinkage}")
    print('-'*42)
    print("SVC Classification accuracy: %f / Chance level: %f" % (np.mean(scores_svc), class_balance))
    print(f"Mean Score Model {mean_scores_svc}")
    print(f"Std Score Model {std_scores_svc}")
    print('-'*24)

    ####################
    # https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html#ex-decoding-csp-eeg
    # Look at performance over time
    ####################
    sfreq = raw.info['sfreq']
    w_length = int(sfreq * 0.5)   # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data_train.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv.split(epochs_data_train):
        print(f"train_idx={train_idx}")
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda_shrinkage.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data_train[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda_shrinkage.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()
    ####################

    lda_shrinkage.fit(csp.fit_transform(epochs_data_train, labels), labels)
    try:
        os.remove('model.joblib')
    except OSError:
        pass
    dump(lda_shrinkage, 'model.joblib')


    # Prediction

    pivot = int(0.5 * len(epochs_data_train))

    clf = clf.fit(epochs_data_train[:pivot], labels[:pivot])
    try :
        p = clf.named_steps["CSP"].plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    except AttributeError:
        print("Method not implemented")

    print("X shape= ", epochs_data_train[pivot:].shape, "y shape= ", labels[pivot:].shape)

    scores = []
    for n in range(epochs_data_train[pivot:].shape[0]):
        pred = clf.predict(epochs_data_train[pivot:][n:n + 1, :, :])
        print("n=", n, "pred= ", pred, "truth= ", labels[pivot:][n:n + 1])
        scores.append(1 - np.abs(pred[0] - labels[pivot:][n:n + 1][0]))
    print("Mean acc= ", np.mean(scores))
    pass


if __name__ == "__main__":
    #DATA_DIR = "mne_data"
    SUBJECTS = [42]
    RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
    RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    RUNS = RUNS2

    # ft_fit()

    # PREDICT_MODEL = "final_model.joblib"
    # SUBJECTS = [2]
    # ft_predict()

    ft_pipeline()
