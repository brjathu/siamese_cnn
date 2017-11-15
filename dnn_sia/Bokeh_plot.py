from bokeh.plotting import figure, output_file, show
from bokeh.models.widgets import Panel, Tabs, Slider, Toggle
from bokeh.layouts import widgetbox, row, column, layout
from bokeh.models import HoverTool, Range1d, CustomJS, ColumnDataSource
import bokeh.plotting as bp
import gensim
from math import log10
import pickle
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models.doc2vec import TaggedDocument
from gensim import models
import scipy.misc
import scipy.io
import random
import numpy as np
import os
from bokeh.models import Legend


loadFromPickle = True

pickleFile = "save.pickle"  # location of pickle file to save t-SNE vectors after calculation
class_list = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

vect_128 = np.load("./data/train_pca.npy")
vect_class = np.load("./data/train_label.npy")[:, 1]


def buildTestSet(fileName):
    # logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Building Dataset")
    X = []
    Y = []
    with open(fileName) as file:
        for i in file:
            c = i.rstrip()
            y = c[-30:]
            x = c[:-30]

            x_list = eval(x)
            y = eval(y)

            X.append(x_list)
            Y.append(y)

    X = np.asarray(X)
    # logEntry(strftime("%Y-%m-%d %H:%M:%S", localtime()) + " **** Done Building Dataset")
    return X, Y


if loadFromPickle:
    vectors = pickle.load(open(pickleFile, "rb"))
    print("Loaded from Pickle")
else:

    print(vect_128.shape)
    print(vect_class.shape)
    print(vect_class)

    tSNEModel = TSNE(n_components=2, random_state=0, metric="cosine", perplexity=50)

    vectors = tSNEModel.fit_transform(vect_128)

    print(vectors.shape)

    print("tSNE done")
    pickle.dump((vectors), open(pickleFile, "wb"))
    # print("saved pickle")

x = [a[0] for a in vectors]
y = [a[1] for a in vectors]


def convert_to_hex(colour):
    red = int(colour[0] * 255)
    green = int(colour[1] * 255)
    blue = int(colour[2] * 255)
    return '#{r:02x}{g:02x}{b:02x}'.format(r=red, g=green, b=blue)


colourClas = [convert_to_hex(cm.gist_rainbow(int(i) * 1.0 / 25)) for i in vect_class]
print(vect_class)

source1 = bp.ColumnDataSource({"Class": vect_class,
                               'x_tSNE': vectors[:, 0],
                               'y_tSNE': vectors[:, 1],
                               'color': colourClas})

output_file("Apps_tSNE.html")

title = "Breath vectors"

p1 = figure(plot_width=1500, plot_height=900, title=title, tools="pan,box_zoom,reset,hover")
p1.title.text_font_size = '25pt'
p1.xaxis.axis_label_text_font_size = "25pt"
p1.yaxis.axis_label_text_font_size = "25pt"


p1.scatter(x='x_tSNE', y='y_tSNE', source=source1, color='color', legend="Class")
p1.legend.location = "bottom_center"
p1.legend.orientation = "horizontal"
p1.legend.label_text_font_size = "25pt"

hover1 = p1.select(dict(type=HoverTool))


htmlLayout = """
        <div style="width:40vw">
            <div>
                <span style="font-size: 15px; color: #966;">@Class</span>
            </div>
        </div>
        """
hover1.tooltips = htmlLayout


tab1 = Panel(child=p1, title="Density")
tabs = Tabs(tabs=[tab1])

show(tabs)
