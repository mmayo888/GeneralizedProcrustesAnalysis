GeneralizedProcrustesAnalysis
=============================

This is an implementation of Generalized Procrustes Analysis (Gower, 1975) for 2D shape data, as a filter for WEKA.

If your examples are in the form of shapes defined by 2D points, and you want to eliminate changes due to rotation, scale and translation, then this filter can be applied to your data prior to passing it to a classifier for machine learning.

Datasets should be prepared in ARFF format such that every example has a fixed number of numeric attributes named x1, y1, x2, y2, x3, y3, ...

For an example of how to prepare the datasets, see the datasets folder. Two of the datasets contain plethodon data from the Geomorph R package (Adams & Otarola-Castillo, 2013).

You may also include non 2D-point attributes in the dataset, and they will simply be ignored by the filter.

Two filters are provided: an unsupervised version that applies GPA to all the data, and a supervised version that applies GPA to data from each class individually, mapping each example that consists of N 2D points onto a new example consisting of NC 2D points where C is the number of classes.

For an example of how to run experiments at the command line, see the run.sh file. Otherwise simply use the filter from the WEKA explorer or experimenter.

References:

Adams, Dean A. and Otarola-Castillo E. 2013. Geomorph: An R package for the collection and analysis of geometric morphometric shape data.

Gower J. 1975. Generalized procrustes analysis. Psychometrika 40(1), pp 33-51.


