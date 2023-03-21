import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from collections import deque

from ft_utils import raw_filenames, fetch_data, prepare_data, filter_data, fetch_events
from ft_fit import ft_fit


if __name__ == "__main__":

    subject = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
                71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                101, 102, 103, 104, 105, 106, 107, 108, 109]

    score   =  [ 1.0, 0.71,  1.0,  1.0, 0.96, 0.93, 0.98,  1.0,  1.0, 0.96,
                0.96,  1.0,  1.0, 0.98,  1.0, 0.98,  1.0,  1.0,  1.0,  1.0,
                 1.0,  1.0, 0.78, 0.93,  1.0,  1.0, 0.98, 0.98, 0.93, 0.56,
                0.91, 0.84, 0.96, 0.96, 0.96,  1.0,  1.0, 0.98, 0.93,  1.0,
                 1.0,  1.0, 0.98, 0.96, 0.98, 1.0,  1.0,  1.0,  0.71, 1.0,
                 1.0, 0.93,  1.0,  1.0, 0.98, 0.98, 0.93, 0.93, 0.91, 0.93,
                 1.0,  1.0,  1.0, 0.96, 0.96, 0.96, 0.87,  1.0, 0.96, 0.93,
                 1.0, 0.96,  1.0, 0.98, 0.87, 0.91,  1.0, 0.71, 0.84,  1.0,
                 0.8, 0.93,  1.0,  1.0,  1.0,  1.0,  1.0, 0.77, 0.96,  1.0,
                0.93, 0.95,  1.0, 0.64, 0.96, 0.82, 0.98,  1.0, 0.98,  1.0,
                 1.0,  1.0,  1.0,  1.0,  1.0, 0.98, 0.89, 0.98, 0.98]

    df = pd.DataFrame({'subject': subject,
                       'score': score})

    mask = (df.score == 1.0)
    df2 = df.loc[mask, :]

    print(df2.head(10))
    print(df2["score"].mean())
    subject_list = df2["subject"].values.tolist()
    print(f'subject_list={subject_list}, len={len(subject_list)}')
    exit()

    df = pd.read_csv('../score.csv')
    print(df.head(10))
    print(df["score"].mean())

    exit()

    #%matplotlib qt5

    #DATA_DIR = "mne_data"

    # RUNS1 = [6, 10, 14]  # motor imagery: hands vs feet
    # RUNS2 = [4, 8, 12]  # motor imagery: left hand vs right hand
    # RUNS = RUNS2
    # plt.ion()

    # SUBJECTS = [1]
    # ft_fit(SUBJECTS, RUNS)

    #plt.ion()
    fig = plt.figure(figsize=(4.2, 4.2))
    plt.plot(range(10), range(10))
    plt.show()
    plt.ion()


    # PREDICT_MODEL = "final_model.joblib"
    # SUBJECTS = [2]
    # ft_predict()

    # ft_pipeline()



