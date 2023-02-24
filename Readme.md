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
#pip list --format=freeze > requirements.txt
#pip install -r requirements.txt

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

eegbci_training+predict0*.ipynb CSPTEST2 vs ms/csp.py

x_class0.shape3=(64, 3703), cov0.shape=(64, 64)
x_class1.shape3=(64, 3542), cov1.shape=(64, 64)
covs.shape=(2, 64, 64), sample_weight=[64 64]
eigenvalues.shape=(64,), eigenvectors.shape=(64, 64)
eigen_values, eigen_vectors = scipy.linalg.eigh(covs[0], covs.sum(0))

eigenvalues=
[0.17746337 0.24118598 0.27384043 0.28964807 0.29520345 0.32522068
 0.33402773 0.34438121 0.35182909 0.3651519  0.37221281 0.37947045
 0.38496811 0.38944195 0.39464295 0.40171966 0.40584041 0.40909791
 0.41352371 0.41914926 0.42786026 0.43452281 0.43670259 0.43925342
 0.44267184 0.44770765 0.45582463 0.46441613 0.46614347 0.47155521
 0.47228989 0.47905544 0.483337   0.48627152 0.49914949 0.50008759
 0.50469975 0.51076627 0.51713878 0.51901223 0.52420346 0.52635209
 0.53216028 0.53694715 0.54490356 0.54741154 0.55257027 0.55919848
 0.56525476 0.57049208 0.57708014 0.57722571 0.58420622 0.58616418
 0.59852949 0.60565039 0.6097799  0.61295291 0.62875051 0.64452457
 0.65106847 0.65522958 0.68470646 0.709407  ], eigenvectors.shape=(64, 64)

np.abs(eigen_values - 0.5)=
[3.22536629e-01 2.58814023e-01 2.26159570e-01 2.10351931e-01
 2.04796545e-01 1.74779322e-01 1.65972268e-01 1.55618788e-01
 1.48170906e-01 1.34848098e-01 1.27787194e-01 1.20529551e-01
 1.15031894e-01 1.10558052e-01 1.05357050e-01 9.82803406e-02
 9.41595863e-02 9.09020887e-02 8.64762874e-02 8.08507425e-02
 7.21397388e-02 6.54771909e-02 6.32974123e-02 6.07465800e-02
 5.73281622e-02 5.22923493e-02 4.41753652e-02 3.55838688e-02
 3.38565306e-02 2.84447888e-02 2.77101107e-02 2.09445638e-02
 1.66630020e-02 1.37284787e-02 8.50507250e-04 8.75943758e-05
 4.69975333e-03 1.07662706e-02 1.71387791e-02 1.90122347e-02
 2.42034625e-02 2.63520938e-02 3.21602820e-02 3.69471454e-02
 4.49035582e-02 4.74115373e-02 5.25702711e-02 5.91984838e-02
 6.52547568e-02 7.04920831e-02 7.70801444e-02 7.72257103e-02
 8.42062173e-02 8.61641753e-02 9.85294874e-02 1.05650393e-01
 1.09779897e-01 1.12952910e-01 1.28750510e-01 1.44524567e-01
 1.51068475e-01 1.55229579e-01 1.84706461e-01 2.09407004e-01]

np.argsort(np.abs(eigen_values - 0.5))=
[35 34 36 37 33 32 38 39 31 40 41 30 29 42 28 27 43 26 44 45 25 46 24 47
 23 22 48 21 49 20 50 51 19 52 53 18 17 16 15 54 14 55 56 13 57 12 11 10
 58  9 59  8 60 61  7  6  5 62  4 63  3  2  1  0]

np.argsort(np.abs(eigen_values - 0.5))[::-1]=
[ 0  1  2  3 63  4 62  5  6  7 61 60  8 59  9 58 10 11 12 57 13 56 55 14
 54 15 16 17 18 53 52 19 51 50 20 49 21 48 22 23 47 24 46 25 45 44 26 43
 27 28 42 29 30 41 40 31 39 38 32 33 37 36 34 35]

ix=
[ 0  1  2  3 63  4 62  5  6  7 61 60  8 59  9 58 10 11 12 57 13 56 55 14
 54 15 16 17 18 53 52 19 51 50 20 49 21 48 22 23 47 24 46 25 45 44 26 43
 27 28 42 29 30 41 40 31 39 38 32 33 37 36 34 35]

eigenvectors2.shape=(64, 64)

vs

X1.shape=(2, 500), S1.shape=(2, 2)
X2.shape=(2, 500), S2.shape=(2, 2)
eigen_values, eigen_vectors =scipy.linalg.eigh(S1, S1+S2)
eigenvalues=[0.1978  0.80471], eigenvectors.shape=(2, 2)
----------------------------------------
```


## to-do list

```
  - Preprocessing
    - being shown in the video
      . men-tools says it provides the functino, but not working.
      . needs to figure out to do it
    - making an additional filter is not necessary
      . The feature extraction section is just to check that
        the significative frequencies for a motor imagery task are
        kept (~8-40Hz)

  - Classification
    - Predict: uses sklearn validation tools?

  - Implementation
    - Score: Over 75% add a point for every 3%. current score is 0

    # 1,   2,     3,    4,     5,   42
    # 0.8, 0.867, 0.73, 0.667, 0.8, 0.867

    # 0.766

```
