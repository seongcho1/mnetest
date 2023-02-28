import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

from collections import deque

from ft_utils import raw_filenames, fetch_data, prepare_data, filter_data, fetch_events
from ft_fit import ft_fit


if __name__ == "__main__":


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



