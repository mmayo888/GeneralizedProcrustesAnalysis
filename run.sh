javac -cp ~/Applications/weka-3-7-9/weka.jar:./ ./weka/filters/unsupervised/instance/*.java
javac -cp ~/Applications/weka-3-7-9/weka.jar:./ ./weka/filters/supervised/instance/*.java

java -cp ~/Applications/weka-3-7-9/weka.jar weka.classifiers.trees.J48 -t datasets/plethodon.arff 
java -cp ~/Applications/weka-3-7-9/weka.jar:./ weka.classifiers.meta.FilteredClassifier -F "weka.filters.unsupervised.instance.GPAFilter2D -S 42 -I 5 -C false" -W weka.classifiers.trees.RandomForest -t datasets/plethodon.arff -- -I 100 
java -cp ~/Applications/weka-3-7-9/weka.jar:./ weka.classifiers.meta.FilteredClassifier -F "weka.filters.supervised.instance.SupervisedGPAFilter2D -S 42 -I 5 -C true" -W weka.classifiers.trees.RandomForest -t datasets/plethodon.arff -- -I 100 