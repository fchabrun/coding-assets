# -*- coding: utf-8 -*-
"""
Created on Mar 30 2020

@author: Floris Chabrun
"""

import numpy as np
from sklearn import metrics
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt

def plotROC(y, y_, thresholds = [dict(value=.5, name="default"),dict(value="youden", name="Youden")], new_figure=True, show=True, invert_if_auc_below_50=False,
            auto_va_alignment=False,
            **kwargs):
    """
    
    Plots the ROC curve with the AUC-ROC
    ====================================
    
    :Example:
        
    # Will display a simple ROC curve, with Se/Sp values for threshold closest to 0.5 and for threshold with best Youden's index
    >>> plotROC(y, y_)
    
    # Generates a random distribution and display a simple ROC curve, with same values displayed plus Se/Sp for the threshold closest to 0.9
    >>> plotROC(np.concatenate((np.zeros((100,)),np.ones((100,)))),
    >>>         np.concatenate((np.random.normal(loc=0.4, scale=.3, size=100),np.random.normal(loc=0.8, scale=.3, size=100))),
    >>>         thresholds = [dict(value=.5, name="default"),
    >>>                       dict(value=.9),
    >>>                       dict(value="youden", name="Youden")])

    Description
    -----------
    
    Plots a ROC curve.
    
    Parameters
    ----------
    
    y_test : numpy.array or pandas.DataFrame
        ground truth vector/array/column
        
    y_test_ : numpy.array or pandas.DataFrame
        predictions vector/array/column
        
    thresholds : list of dict
        thresholds for which sensitivity and specificity should be displayed.
        Each threshold should be a dict containing at least a "value" key,
        either a numerical value representing the desired threshold,
        or "Youden" for determining the best threshold through Youden's index.
        If a numerical value is set, the threshold closest to this value will be chosen.
        The dict can contain an optional "name" key which will be displayed along with
        the threshold
    
    new_figure : bool, default=True
        whether or not to create a new figure. Set this value to `False` to plot into an existing figure
        
    title : str, optional
        the title of the plot
        
    invert_if_auc_below_50 : bool, default=False
        if True, y will be inverted if auc is lower than 0.5
        
    """
    if invert_if_auc_below_50:
        tmp_auc = metrics.roc_auc_score(y, y_)
        if tmp_auc < .5:
            if type(y)==list:
                y = [1-yi for yi in y]
            else:
                y = 1-y
        
    fpr, tpr, t = metrics.roc_curve(y, y_)
    roc_auc = metrics.auc(fpr, tpr)
    
    if new_figure:
        plt.figure(figsize=(8,8))

    # Plot curve
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    bbox_props = dict(fc="white")

    t_i_lower, t_i_higher = -1, -1
    ll_thresh, hh_thresh = np.Inf, -np.Inf
    if auto_va_alignment:  # compute position
        for t_i, threshold in enumerate(thresholds):
            # select indice among possible thresholds
            if threshold["value"] == "youden":
                youden = (1-fpr)+tpr-1
                where=np.where(youden == np.max(youden))[0][0]
            else:
                where=np.argmin(np.abs(t-threshold["value"]))
            y_height_here = tpr[where]
            if y_height_here < ll_thresh:
                t_i_lower = t_i
                ll_thresh = y_height_here
            if y_height_here > hh_thresh:
                t_i_higher = t_i
                hh_thresh = y_height_here
        if (t_i_lower == t_i_higher) and (len(thresholds) > 1):
            t_i_lower = 0
            t_i_higher = 1

    # Display thresholds
    for t_i, threshold in enumerate(thresholds):
        # select indice among possible thresholds
        if threshold["value"] == "youden":
            youden = (1-fpr)+tpr-1
            where=np.where(youden == np.max(youden))[0][0]
        else:
            where=np.argmin(np.abs(t-threshold["value"]))
        # plot a point on the curve
        plt.plot(fpr[where], tpr[where], 'o', color = 'orange')
        # display text
        if t_i == t_i_lower:
            va = "top"
        elif  t_i == t_i_higher:
            va = "bottom"
        else:
            va = "center"
        if "name" in threshold.keys() and threshold["name"] is not None and len(threshold["name"])>0:
            plt.text(fpr[where]+.02,
                     tpr[where],
                     'Threshold={:.2E} ({}): Se={:.1f}%, Sp={:.1f}%'.format(t[where], threshold["name"], 100*tpr[where], 100*(1-fpr[where])),
                     bbox = bbox_props,
                     verticalalignment=va,
                     wrap = True)
        else:
            plt.text(fpr[where]+.02,
                     tpr[where],
                     'Threshold={:.2E}: Se={:.1f}%, Sp={:.1f}%'.format(t[where], 100*tpr[where], 100*(1-fpr[where])),
                     bbox = bbox_props,
                     verticalalignment=va,
                     wrap = True)

    # Set boundaries, labels, titles...
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    if "title" in kwargs.keys():
        plt.title(kwargs['title'])
    plt.legend(loc="lower right")
    if show:
        plt.show()
    
def quickBoxPlot(y, x, plot_mean=True, ttest=True):
    distributions = []
    y = np.array(y).copy()
    x = np.array(x).copy()
    x = x[np.isnan(y)==False]
    y = y[np.isnan(y)==False]
    y = y[np.isnan(x)==False]
    x = x[np.isnan(x)==False]
    if x.shape != y.shape:
        raise Exception("Values (x) and annotations (y) must be the same length")
    for y_value in np.unique(y):
        distributions.append(dict(name="{}".format(y_value), values=x[y==y_value]))
    
    y_max = np.max(np.concatenate([distrib["values"] for distrib in distributions]))
    y_min = np.min(np.concatenate([distrib["values"] for distrib in distributions]))
    y_unit = (y_max-y_min)/10
    
    
    plt.figure()
    plt.boxplot([distrib["values"] for distrib in distributions])
    plt.xticks(ticks = list(range(1,len(distributions)+1)), labels=[distrib["name"] for distrib in distributions])
    if plot_mean:
        for i,distrib in enumerate(distributions):
            mean_v = np.mean(distrib["values"])
            plt.text(i+.5,mean_v,"Mean: {:.1f}".format(mean_v))
    if ttest:
        for level in range(1,len(distributions)):
            for i in range(len(distributions)-level):
                ie = i+level
                # compute p-value
                p=ttest_ind(distributions[i]["values"], distributions[ie]["values"]).pvalue
                # plot text
                plt.plot([i+1.02,i+1.02,ie+.98,ie+.98], [y_max+(level-.1)*y_unit,y_max+level*y_unit,y_max+level*y_unit,y_max+(level-.1)*y_unit])
                plt.text((i+ie+2)/2, y_max+(level+.1)*y_unit, "p={:.1E}".format(p), horizontalalignment="center")
    plt.show()
    
def ezBoxPlot(distributions, ttest=True):
    """
    
    Easy box plotting and t-testing of samples distributions
    ========================================================
    
    :Example:
        
    # Simply compare two distributions
    >>> ezBoxPlot(distributions=[dict(name="cases",values=x1),
    >>>                          dict(name="controls",values=x2)])
    # Compare multiple distributions
    >>> ezBoxPlot(distributions=[dict(name="young_cases",values=x1),
    >>>                          dict(name="young_controls",values=x2),
    >>>                          dict(name="old_cases",values=x3),
    >>>                          dict(name="old_controls",values=x4)])
    
    Description
    ----------
    
    Parameters
    ----------
    
    distributions : list of dict
        The distributions to plot and compare. Should be similar to : [dict(name="case",values=x1),dict(name="controls",values=x2),...]
        
    ttest : bool, default=True
        whether to compute and display t-test p-values between distributions
    
    """
#    distributions=[dict(name="cases",values=x1),
#                   dict(name="controls",values=x2),
#                   dict(name="new_cases",values=x1),
#                   dict(name="new_controls",values=x2)]
    
    y_max = np.max(np.concatenate([distrib["values"] for distrib in distributions]))
    y_min = np.min(np.concatenate([distrib["values"] for distrib in distributions]))
    y_unit = (y_max-y_min)/10
    
    
    plt.figure()
    plt.boxplot([distrib["values"] for distrib in distributions])
    plt.xticks(ticks = list(range(1,len(distributions)+1)), labels=[distrib["name"] for distrib in distributions])
    if ttest:
        for level in range(1,len(distributions)):
            for i in range(len(distributions)-level):
                ie = i+level
                # compute p-value
                p=ttest_ind(distributions[i]["values"], distributions[ie]["values"]).pvalue
                # plot text
                plt.plot([i+1.02,i+1.02,ie+.98,ie+.98], [y_max+(level-.1)*y_unit,y_max+level*y_unit,y_max+level*y_unit,y_max+(level-.1)*y_unit])
                plt.text((i+ie+2)/2, y_max+(level+.1)*y_unit, "p={:.1E}".format(p), horizontalalignment="center")
    plt.show()

    
if __name__ == "__main__":
    plotROC(np.concatenate((np.zeros((100,)),np.ones((100,)))),
            np.concatenate((np.random.normal(loc=0.4, scale=.3, size=100),np.random.normal(loc=0.8, scale=.3, size=100))),
            thresholds = [dict(value=.5, name="default"),
                          dict(value=.9),
                          dict(value="youden", name="Youden")])
