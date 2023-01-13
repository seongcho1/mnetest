import os
from copy import deepcopy
import numpy as np
import mne

#sample_data_folder = mne.datasets.sample.data_path('./')

sample_data_raw_file = ('./data/MNE-sample-data/MEG/sample/sample_audvis_filt-0-40_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file)

print(raw)
print(raw.info)

#raw.plot_psd(fmax=50)
#spectrum = raw.compute_psd()
#spectrum.plot(average=True)
#raw.plot(duration=5, n_channels=30)
#raw.plot()
print(raw.info['bads'])

picks = mne.pick_channels_regexp(raw.ch_names, regexp='EEG 05.')
print(picks)
#raw.plot(order=picks, n_channels=len(picks))


picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG 2..3')
print(picks)
raw.plot(order=picks, n_channels=len(picks))
