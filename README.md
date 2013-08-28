GeneralizedProcrustesAnalysis
=============================

This is an implementation of Generalized Procrustes Analysis for 2D shape data, as a filter for WEKA.

If your examples are in the form of shapes defined by 2D points, and you want to eliminate changes due to rotation, scale and translation, then this filter can be applied to your data prior to passing it to a classifier for machine learning.

Datasets should be prepared in ARFF format such that every example has a fixed number of numeric attributes named x1, y1, x2, y2, x3, y3, ...

For an example of how to prepare the datasets, see the datasets folder.

Two filters are provided: an unsupervised version that applies GPA to all the data, and a supervised version that applies GPA to data from each class individually, mapping each example that consists of n points onto a new example consisting of nc points where nc is the number of classes.

For an example of how to run experiments at the command line, see the run.sh file. 
