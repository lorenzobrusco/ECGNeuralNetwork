## Local Installation

### Clone the repo
```shell
$ git clone https://github.com/mtobeiyf/keras-flask-deploy-webapp.git
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
$ python app.py
```
