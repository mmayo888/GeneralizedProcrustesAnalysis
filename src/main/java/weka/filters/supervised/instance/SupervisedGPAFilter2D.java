/*
 * Supervised GPA Filter for 2D shape data
 *
 * by
 * Michael Mayo, 2013
 *
 */


package weka.filters.supervised.instance;

import java.util.*;

import weka.core.*;
import weka.filters.*;

import weka.filters.unsupervised.instance.GPAFilter2D;

public class SupervisedGPAFilter2D extends GPAFilter2D  {

    // Number of classes
    private int numClasses=-1;
    
    // An array of GPAFilters, one for each class
    private GPAFilter2D[] filters;
    
    // Random Number generator
    private Random rng;
    
    // Return info about the filter
    public String globalInfo() {
        return "Supervised Wrapper for Generalized Procrustes Analysis.";
    }
      

    // The filter does change the format of the data: specifically it makes one copy of the attributes per class
    protected Instances determineOutputFormat(Instances inputFormat) {
        return process(inputFormat);
    }
   
    // The filter applies to any dataset and does not require a class attribute
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enableAllClasses();  
        return result;
    }
  
    // Setting the options
    public void setOptions(String[] options) throws Exception{
        super.setOptions(options);
    }
    
    // Getting the options
    public String[] getOptions() {
        return super.getOptions();
    }
    
    // Enumeration describing all the options
    public Enumeration listOptions() {
        return super.listOptions();
    }
    
    
    // Create one filter for each class
    public Instances process(Instances data) {
        if (numClasses==-1) {
            // First time filter has run, so do some setup
            rng=new Random( getSeed() );
            // Create GPA filter for each class
            numClasses=data.numClasses();
            filters = new GPAFilter2D[ numClasses ];
            for (int classIndex=0; classIndex<numClasses; classIndex++) {
                // Perform GPA on the point data
                filters[ classIndex ] = new GPAFilter2D();
            }
            // Subset the data by class
            for (int classIndex=0; classIndex<numClasses; classIndex++) {
                // Select all examples beloning to the current class
                Instances subset = new Instances(data,0);
                Attribute classAtt = data.classAttribute();
                for (Instance instance: data) {
                    if (instance.value(classAtt)==classIndex)
                        subset.add(instance);
                }
                // Set up the filter
                filters[ classIndex ].setSeed( rng.nextInt() );
                filters[ classIndex ].setNumIterations( getNumIterations() );
                filters[ classIndex ].setAllowScaling( getAllowScaling() );
                // Process the examples for this class only
                filters[ classIndex ].process( subset );
                
                
            }
            // Done! We now have one filter per class set up
        }
        // Now we construct a new dataset by merging the filtered output of each filter
        Instances[] filteredData = new Instances[ numClasses ];
        for (int classIndex=0; classIndex<numClasses; classIndex++) {
            // Create the inidividual datasets
            filteredData[ classIndex ] = new Instances( filters[ classIndex ].process(data) );
            // Make the attribute names unique
            for (int attIndex=0; attIndex<filteredData[ classIndex ].numAttributes(); attIndex++) {
                Attribute attribute = filteredData[ classIndex ].attribute(attIndex);
                filteredData[ classIndex ].renameAttribute(attribute, "class"+classIndex+"_"+attribute.name());
            }
            // For all but the last dataset, remove  the class variable
            if (classIndex != (numClasses-1) ) {
                // Remove the class attribute
                int attToDelete = filteredData[ classIndex ].classIndex();
                filteredData[ classIndex ].setClassIndex(-1);
                filteredData[ classIndex ].deleteAttributeAt( attToDelete );
            }
        }
        // Merge the datasets
        Instances merged=filteredData[ 0 ];
        for (int classIndex=1; classIndex<numClasses; classIndex++) {
            merged = Instances.mergeInstances( merged, filteredData[ classIndex ] );
        }
        // *** TODO: If there is a weka filter for removing duplicate attributes, apply it now to the merged data
        // Set the class to whatever the class of the last dataset copy was
        merged.setClass( filteredData[ numClasses-1 ].classAttribute() );
        return merged;
    }
  
     
    
  

  
}