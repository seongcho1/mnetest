# EEG Motor Movement/Imagery
https://www.slideshare.net/DonghyeonKim7/eeg-102812945

https://www.mdpi.com/2079-9292/11/15/2293/pdf

# Data

https://archive.physionet.org/pn4/eegmmidb/


https://physionet.org/content/eegmmidb/1.0.0/


https://github.com/mne-tools/mne-python/blob/main/mne/datasets/eegbci/eegbci.py


https://braindecode.org/stable/auto_examples/plot_mne_dataset_example.html#sphx-glr-auto-examples-plot-mne-dataset-example-py


# Tutorials


https://mne.tools/1.0/auto_tutorials/index.html


https://mne.tools/1.0/auto_examples/decoding/decoding_csp_eeg.html


https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/Introduction.html#Algorithms-in-neuroscience


https://github.com/jona-sassenhagen/mne_workshop_amsterdam


https://www.youtube.com/watch?v=IYuAPisoUeI&list=PLXtvZiGkmNVvPS0N9UNBVkIFe0_0t_Nqt


https://github.com/NeuroTechX/moabb


https://github.com/lkorczowski/BCI-2021-Riemannian-Geometry-workshop/blob/master/notebooks/MOABB-approach.ipynb


https://www.kaggle.com/code/shoolpani/eye-blink-detection-using-eegmmidb/notebook?scriptVersionId=59041125


# References

http://learn.neurotechedu.com/machinelearning/#machine-learning-for-brain-computer-interfaces


https://pyriemann.readthedocs.io/en/latest/auto_examples/motor-imagery/plot_single.html


https://github.com/JulesBelveze/eeg-classifier/tree/master/classification


with PyTorch, motor imagery: hands vs feet

https://github.com/mne-tools/mne-torch/blob/master/demo_eeg_csp.py

with sklearn, motor imagery: hands vs feet

https://gist.github.com/F-A/ebd0cf72fb4e8b43d6d278db776f5824


EEG Feature Extraction

https://www.youtube.com/watch?v=rgG9t6DrBAk



https://www.youtube.com/watch?v=AjMdirPPnQQ


https://docs.google.com/presentation/d/1KHbTb6H09P7SWbL6a8ryK1QKINZb5FbTMNx-DMoqaEY/edit#slide=id.p

# References2

https://www.sciencedirect.com/science/article/pii/S1110016821007055


https://arro.anglia.ac.uk/id/eprint/706861/1/Selim_2021.pdf


https://www.slideshare.net/victorasanza/eeg-signal-clustering-for-motor-and-imaginary-motor-tasks-on-hands-and-feet-85935652



https://www.google.com/search?rlz=1C9BKJA_enKR930KR930&hl=en-US&sxsrf=AJOqlzWHUsvpLbaAMpi0ZmiND_8S8lDyDA:1674139424923&q=EEG+motor+imagery-classification+GitHub&sa=X&ved=2ahUKEwj2iuit79P8AhVi8zgGHYWOAGUQ1QJ6BAgiEAE&biw=1080&bih=695&dpr=2




# Question
[total-perspective-vortex] In the subject, page5, first paragraph, second sentence.
https://cdn.intra.42.fr/pdf/pdf/60845/en.subject.pdf

The data was measured during a motor imagery experiment, where people had to do or imagine a hand or feet movement.

I am wondering if what you meant was using only imagery data.

The data was measured during a motor imagery experiment, where people had to imagine a hand or feet movement.


# shared

Bandpass filter Burttherworth-IIR, 7-30 Hz from page 7 of https://www.slideshare.net/victorasanza/eeg-signal-clustering-for-motor-and-imaginary-motor-tasks-on-hands-and-feet-85935652

subject -> subjects from [eeg_motor_imagery_002_Motor_imagery_decoding_from_EEG_data_using_the_Common_Spatial_Pattern_(CSP).ipynb](https://github.com/seongcho1/mnetest/blob/main/eeg_motor_imagery_002_Motor_imagery_decoding_from_EEG_data_using_the_Common_Spatial_Pattern_(CSP).ipynb)

64 channels -> motor cortex area (7 x 3 channels) from [pdf/02The EEG Device For Your Project_ Choosing between NeuroSky MindWave, Muse 2, g.tec Unicorn, OpenBCI, Emotiv EPOC+… _ by Tim de Boer _ A Beginner’s Guide to Brain-Computer Interfaces _ Medium.pdf](https://medium.com/the-ultimate-bedroom-bci-guide/a-beginners-guide-to-brain-computer-interfaces-part-2-how-to-choose-the-eeg-device-for-your-eb8d51fa5d66)

Common Average Referencing (CAR) data -= data.mean() from [pdf/06Improving Preprocessing Of EEG Data In One Line Of Code With CAR _ by Tim de Boer _ A Beginner’s Guide to Brain-Computer Interfaces _ Medium.pdf](https://medium.com/the-ultimate-bedroom-bci-guide/improving-preprocessing-of-eeg-data-in-one-line-of-code-with-car-a9f7cc52e3fc)


# data info


```
#https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py

# raw = filter_data(raw=prepare_data(raw=fetch_data(raw_fnames=raw_filenames())))
# labels, epochs = fetch_events(raw)

print(epochs.tmin, epochs.tmax)
#-1.0 4.0

#Read epochs (train will be done only between 1 and 2)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.) #-1~4 to 1~2
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

print(labels.shape)
#(45,)
print(labels)
# [0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 1 1 0 1 0 1 0 1 0 1 0 0 1 0 0 1 1 0 1 0 0 1 1 0 1 0 0 1 0]

print(type(epochs))
#<class 'mne.epochs.Epochs'>
print(type(epochs_data))
#<class 'numpy.ndarray'>
print(epochs_data.shape)

#data for 45 events and 801 original time points ...
#45 matching events found
#64-channel EEG signals
#801 original time points

#(45, 64, 801)

print(type(epochs_train))
#<class 'mne.epochs.Epochs'>
print(type(epochs_data_train))
#<class 'numpy.ndarray'>
print(epochs_data_train.shape)
#(45, 64, 161)

#epochs_data_train

45개 events가 class0 23개, class1 22개로 나뉨
----------------------------------------
_concat_cov: x_class.shape0=(23, 64, 161)
_concat_cov: x_class.shape1=(64, 23, 161)
_concat_cov: x_class.shape2=(64, 3703)
_concat_cov: x_class.shape3=(64, 3703), x_class.T.shape=(3703, 64), x_class.T.conj().shape=(3703, 64)
----------------------------------------
_concat_cov: x_class.shape0=(22, 64, 161)
_concat_cov: x_class.shape1=(64, 22, 161)
_concat_cov: x_class.shape2=(64, 3542)
_concat_cov: x_class.shape3=(64, 3542), x_class.T.shape=(3542, 64), x_class.T.conj().shape=(3542, 64)
----------------------------------------

x_class0.shape3=(64, 3703), cov0.shape=(64, 64)
x_class1.shape3=(64, 3542), cov1.shape=(64, 64)

vs

X1.shape=(2, 500), S1.shape=(2, 2)
X2.shape=(2, 500), S2.shape=(2, 2)

----------------------------------------
```
