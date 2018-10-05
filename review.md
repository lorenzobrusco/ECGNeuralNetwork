## ECG-Arrhythmia-classification
During this work I was inspired from this paper https://arxiv.org/pdf/1804.06812.pdf in which they classify ECG into seven categories, one being normal and the other six being different types of arrhythmia using convolutional neural network.
Using a convolutional neural networks with 11 layers I've reached an accuracy equal to 94.01%, this result has been achieved adding some data augmentation.

### Convert Signal to Image

Convolutional neural networks required images as input, therefore, I transformed signals into ECG images by plotting each ECG beat as an individual 128 x 128 grayscale image.

MIT-BIH provides an additional file containing information about Q-wave peak.

Thus we create image for each Q-wave using thi formula:



<div style="text-align:center">
    <img src ="media/formula.png"/>
</div>





<div style="text-align:center">
    <img src ="media/signal_to_img.png"/>
</div>


### Classes labels

As a result, we obtained 100,000 images from the MIT-BIH arrhythmia
database where each image is one of eight ECG beat types.

<div style="text-align:center">
    <img src ="media/beats.png"/>
</div>



### Data Augmentation

Data augmentation is one of the key benefits of using images as input data.
The majority of previous ECG arrhythmia works could not manually add an data into training set since the distortion of single ECG signal may downgrade the performance in the test set.

Transformation:

- Rotation (-30°, +30°)
- Flip (Horizontal, Vertical, Horizontal-Vertical)
- Cropping



<div style="text-align:center">
    <img src ="media/augmentation.png"/>
</div>


### Convolutional Neural Networks Struction

<div style="text-align:center">
    <img src ="media/layers.png"/>
</div>


## Model

Here is the link to the model: https://drive.google.com/open?id=12Hk4F6VDEeCahq7IMeS8j2Sbhf1Hsq9k



## Result

The ECG arrhythmia recordings used in this paper are obtained from the MITBIH database. 

The database contains 48 half-hour ECG recordings collected from 47 patients between 1975 and 1979. 

The ECG recording is sampled at 360 samples per second. There are approximately 110,000 ECG beats in MIT-BIH database with 15 different types of arrhythmia including normal.

Dataset has been divided i three parts:

- Training set (70%)
- Validation set (15%)
-  Test set (15%)

The network has been traindend for 100 epochs for 10 hours.

### Training graphs

In the graphs below we see that the network, probably, could continue to improve.

<div style="text-align:center">
    <img src ="media/result-train.png"/>
</div>



### Confusion matrix (Test set)

|         | VEB          | RBB          | NOR          | VFW      | APC       | PAB          | PVC           | LBB           |
| ------- | ------------ | ------------ | ------------ | -------- | --------- | ------------ | ------------- | ------------- |
| **VEB** | **1.196,00** | 7,00         | 0,00         | 1,00     | 0,00      | 2,00         | 33,00         | 10,00         |
| **RBB** | 7,00         | **4.856,00** | 33,00        | 2,00     | 0,00      | 0,00         | 0,00          | 81,00         |
| **NOR** | 0,00         | 42,00        | **1.148,00** | 0,00     | 0,00      | 0,00         | 0,00          | 0,00          |
| **VFW** | 0,00         | 0,00         | 0,00         | **8,00** | 0,00      | 0,00         | 0,00          | 0,00          |
| **APC** | 0,00         | 0,00         | 0,00         | 0,00     | **49,00** | 0,00         | 0,00          | 5,00          |
| **PAB** | 2,00         | 0,00         | 0,00         | 0,00     | 0,00      | **1.043,00** | 45,00         | 0,00          |
| **PVC** | 126,00       | 0,00         | 0,00         | 0,00     | 0,00      | 150,00       | **11.024,00** | 0,00          |
| **LBB** | 55,00        | 343,00       | 0,00         | 5,00     | 15,00     | 0,00         | 0,00          | **11.024,00** |



### Comparison with other paper

| Author       | Acc(%)  |
| ------------ | ------- |
| **Proposed** | **94**  |
| OMsignal     | 92 - 95 |
| Xiong1       | 82      |
| Pyakillya    | 86      |
| İnanGüler    | 96.94   |



### Conclusion 

So far the best result achieved is **acc: 94,01%** and **loss = 0.38**



## Local Installation

### Clone the repo
```shell
$ git clone https://github.com/lorenzobrusco/ECGNeuralNetwork.git
```




### Download dataset

```shell
wget --mirror --no-parent https://www.physionet.org/physiobank/database/mitdb/
```



### Install requirements

```shell
$ pip install -r requirements.txt
```

Make sure you have the following installed:
- Werkzeug
- Flask
- numpy
- Keras
- gevent
- pillow
- h5py
- tensorflow
- opencv-python
- biosppy
- wfdb
- tqdm




### Run with Python

Python 2.7 or 3.5+ are supported and tested.

```shell
$ python ecgnn.py
```



### Create Images

Run this script to create the dataset
```shell
$ python convert_signal_img.py
```
