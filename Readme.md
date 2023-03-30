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


----------------------

# eeg 뇌파

## 뇌파
https://ko.wikipedia.org/wiki/뇌파

## 뉴런과 시냅스
https://blog.naver.com/msnayana/80142231659

신경세포 사이에 시냅스가 형성되어 신경전달물질을 분비, 이온채널에 부착, 이온채널 열림
https://www.etri.re.kr/webzine/20170630/sub04.html

## markov blanket
https://en.wikipedia.org/wiki/Markov_blanket


In statistics and machine learning, when one wants to infer a random variable with a set of variables, usually a subset is enough, and other variables are useless. Such a subset that contains all the useful information is called a Markov blanket.



# 데이터 전처리 - 차원축소

https://chobokim.tistory.com/29?category=951467
feature selection vs feature extraction

https://www.sciencedirect.com/topics/engineering/common-spatial-pattern

https://en.wikipedia.org/wiki/Common_spatial_pattern
CSP corresponds to Principal component analysis

# src/ms/csp.py

```
# 1. Compute the covariance matrix of each class
S1=np.cov(X1)
S2=np.cov(X2)

# 2. Solve the eigenvalue problem S1·W = l·S2·W
l,W=LA.eigh(S1, S1+S2)
l = np.round(l, 5)
A=(np.linalg.inv(W)).T

# 3. get CSP
X1_CSP=np.dot(W.T,X1)
X2_CSP=np.dot(W.T,X2)
```

# src/ft_csp.py

```
class FT_CSP(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
	def transform(self, X):
	def fit_transform(self, X, y, **fit_params):
```

# data length, baseline_correction

src/ft_fit.py
```
if __name__ == "__main__":

    tmin = -0.2  # start of each epoch (200ms before the trigger)
    tmax = 0.5  # end of each epoch (500ms after the trigger)

```

src/ft_utils.py
https://neuro.inf.unibe.ch/AlgorithmsNeuroscience/Tutorial_files/BaselineCorrection.html

```
def fetch_events(data_filtered, tmin=-0.2, tmax=0.5):
    epochs = mne.Epochs(data_filtered, events, event_ids, tmin, tmax, proj=True,
                    picks=picks,
                    baseline=(-0.1, 0),
                    preload=True)

    # epochs_wo_bc = cp.deepcopy(epochs)
    # inteval = (-0.1, 0)
    # bc_epochs = epochs.apply_baseline(inteval)
```

# pipeline - long vs short

src/ft_fit.py
```
def ft_fit(SUBJECTS, RUNS, tmin=-0.2, tmax=0.5, forceplot=False, pipe_long=True):

    lda_shrinkage = LDA(solver='lsqr', shrinkage='auto')
    csp = FT_CSP(n_components=4, reg=None, log=True, norm_trace=False)

	clf = Pipeline([('CSP', csp), ('LDA', lda_shrinkage)])
    clf2 = Pipeline([('LDA', lda_shrinkage)])

```

# dimensionality reduction

```
# https://hal.science/hal-00602686/document
ix = np.argsort(np.abs(eigen_values - 0.5))[::-1]
eigen_vectors = eigen_vectors[:, ix]

self.filters_ = eigen_vectors.T
n_components=4
pick_filters = self.filters_[:self.n_components]
```



