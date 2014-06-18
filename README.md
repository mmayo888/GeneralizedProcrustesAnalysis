GeneralizedProcrustesAnalysis
=============================

This is an implementation of Generalized Procrustes Analysis (Gower, 1975) for 2D shape data, as a filter for WEKA.

If your examples are in the form of shapes defined by 2D points, and you want to eliminate changes due to rotation, scale and translation, then this filter can be applied to your data prior to passing it to a classifier for machine learning.

Datasets should be prepared in ARFF format such that every example has a fixed number of numeric attributes named x1, y1, x2, y2, x3, y3, ...

For an example of how to prepare the datasets, see the datasets folder. The following example datasets are included:

* plethodon-species and plethodon-site: two of the datasets from the Geomorph R package (Adams & Otarola-Castillo, 2013).

* triangles: a dataset of triangles with random locations, orientations and sizes; one class consists of isosceles triangles and the others are equilaterals

* triangles2: another dataset of triangles with random locations and orientations, but this this time all the triangles are equilaterals and the problem is to distinguish large vs small triangles (which means that the scaling option should be turned off when filtering this dataset).

You may also include non 2D-point attributes in the dataset, and they will simply be ignored by the filter and passed on unmodified to the base classifier.

Two filters are provided:

* an unsupervised version that applies GPA to all the data, and 

* a supervised version that applies GPA to data from each class individually, i.e. it maintains a seperate mean shape for each class

This filter should be used in conjunction with the FilteredClassifier, e.g.

java -cp $WEKA_HOME/weka.jar:. weka.classifiers.meta.FilteredClassifier
     -F  "weka.filters.unsupervised.instance.GPAFilter2D -S 42 -I 5 -C false"
     -W  weka.classifiers.trees.RandomForest -t datasets/plethodon-species.arff -- -I 100 

Example of a 10x10 cross validation experiment performed using the WEKA experimenter:

```
Dataset                   (1) bayes.Na | (2) meta.F (3) meta.F
--------------------------------------------------------------
plethodon-site           (100)   70.50 |   100.00 v    88.25 v
plethodon-species        (100)   57.25 |    80.25 v    67.75  
triangles                (100)   51.93 |   100.00 v    78.47 v
triangles2               (100)   56.33 |    51.00     100.00 v
--------------------------------------------------------------
                               (v/ /*) |    (3/1/0)    (3/1/0)


Key:
(1) bayes.NaiveBayes '' 5995231201785697655
(2) meta.FilteredClassifier '-F \"unsupervised.instance.GPAFilter2D -S 42 -I 5 -C true\" -W bayes.NaiveBayes --' -4523450618538717400
(3) meta.FilteredClassifier '-F \"unsupervised.instance.GPAFilter2D -S 42 -I 5 -C false\" -W bayes.NaiveBayes --' -4523450618538717400
```


References:

Adams, Dean A. and Otarola-Castillo E. 2013. Geomorph: An R package for the collection and analysis of geometric morphometric shape data.

Gower J. 1975. Generalized procrustes analysis. Psychometrika 40(1), pp 33-51.


