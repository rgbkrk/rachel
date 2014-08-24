# Zap Rachel

Entry for the Zap Rachel contest at DEF CON 22.

## Prerequisites

You'll need some of the scientific python stack for this, particularly:

* numpy
* pandas
* scikit-learn
* scipy
* matplotlib

Assuming you have the binary dependencies (e.g. `apt-get build-dep python-scipy python-numpy python-matplotlib`), just

```
pip install -r requirements.txt
```

## Analysis

Our [IPython notebook is available on the Notebook Viewer](http://nbviewer.ipython.org/github/rgbkrk/rachel/blob/master/Rachel%20the%20Robo%20Caller.ipynb).

## Run through the original data set

```
python rachel.py
```

## Analysis Machine

To bring up an environment with the notebooks and python files in one IPython notebook package, run this in Docker.

```
docker run -it -e "PASSWORD=pickyourownpasswordnotyournose" -d -p 8888:8888 probablyrgbkrk/collaborachel
```

Then browse to https://127.0.0.1:8888.

