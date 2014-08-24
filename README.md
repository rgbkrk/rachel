# Zap Rachel

Entry for the Zap Rachel contest at DEF CON 22.

## Prerequisites

You'll need some of the scientific python stack for this, particularly:

* numpy
* pandas
* scikit-learn
* scipy

Assuming you have the binary dependencies, just

```
pip install -r requirements.txt
```

## Analysis

Our [IPython notebook is available on the Notebook Viewer](http://nbviewer.ipython.org/github/probablyrgbkrk/robocaller/blob/master/enrich.ipynb).

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

