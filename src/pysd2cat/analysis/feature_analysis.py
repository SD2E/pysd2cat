import sys
import os
import json
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from pprint import pprint
from pysd2cat.data import pipeline
from matplotlib.ticker import NullFormatter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

####
#Important columns: FSC-H, FSC-W

def axis_formatter(ax,x,y,c):
    ax.scatter(x, y, c=c)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    return ax


def tsne_analysis(df,x_colname='FSC-H',y_colname='FSC-W',label_name='class_label'):
    '''
    Run a tsne analysis with multiple perplexities to see what the output data looks like
    :param df: dataframe to perform T-SNE on
    :param label_name: the name of the column that has the label
    :return: nothing, just a T-SNE plot
    '''
    perplexities = [2,5,30,50,100]

    #Choose three subplots, one for both, one for live, and one for dead
    (fig, subplots) = plt.subplots(1, 6, figsize=(15, 8), squeeze=False)
    X = df.drop(columns=['class_label'])
    y = df['class_label'].astype(int)
    green = y == 0
    red = y == 1

    #Define your subplot region
    #BOTH
    ax = subplots[0,0]
    ax = axis_formatter(ax,X[x_colname][red],X[y_colname][red],c="r")
    ax = axis_formatter(ax, X[x_colname][green],X[y_colname][green], c="g")
    ax.set_title("Channels: " + x_colname + " x " + y_colname )
    plt.axis('tight')

    for i, perplexity in enumerate(perplexities):
        ax = subplots[0][i+1]
        print("Running loop {0}: t-sne with perplexity {1}".format(i,perplexity))
        tsne = TSNE(n_components=2, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        ax.set_title("Perplexity=%d" % perplexity)
        ax = axis_formatter(ax,Y[red.as_matrix(),0],Y[red.as_matrix(),1],c="r")
        ax = axis_formatter(ax,Y[green.as_matrix(),0],Y[green.as_matrix(),1],c="g")
        ax.axis('tight')

    print("Saving figure...")
    plt.savefig("Tsne_plot_live_dead.png")
    plt.close()


def clustering_analysis(df,x_colname='FSC-H',y_colname='FSC-W',label_name='class_label'):
    # normalize dataset for easier parameter selection
    #X = StandardScaler().fit_transform(X)
    return 0




def main():
    ## Where data files live
    ##HPC
    data_dir = '/work/projects/SD2E-Community/prod/data/uploads/'

    ##Jupyter Hub
    # data_dir = '/home/jupyter/sd2e-community/'


    print("Building Live/Dead Control Dataframe...")
    live_dead_df = pipeline.get_dataframe_for_live_dead_classifier(data_dir,fraction=.06)
    nrows = len(live_dead_df)
    ncols = len(live_dead_df.columns)
    print("Dataframe constructed with {0} rows and {1} columns".format(nrows,ncols))
    print("Starting t-sne analysis:")
    live_dead_df = live_dead_df.sample(n=1000)
    tsne_analysis(live_dead_df)

if __name__ == '__main__':
    main()
    ###TESTING WITH SCI-KIT DATA####
    #X, y = datasets.make_circles(n_samples=300, factor=.5, noise=.05)
    #df = pd.DataFrame(np.column_stack((X,y)),columns=['col_'+ str(i) for i in range(0,X.shape[1]+1)])
    #df = df.rename(columns={'col_'+str(X.shape[1]):'class_label'})
    #tsne_analysis(df,x_colname='col_0',y_colname='col_1')
