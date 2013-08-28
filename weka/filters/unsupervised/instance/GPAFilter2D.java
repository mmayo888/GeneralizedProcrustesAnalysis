package weka.filters.unsupervised.instance;

import java.util.*;

import weka.core.*;
import weka.filters.*;

public class GPAFilter2D extends SimpleBatchFilter implements Randomizable {


    // Number of points per example (-1 indicates that "process" has not yet been called)
    private int numPoints = -1;
  
    // A reference to the header of the training data
    private Instances trainingDataHeader;
  
    // The random number seed
    private int seed=42;
  
    // Random number generator used to select the reference
    private Random rng;
    
    // Randomly selected reference example
    private Instance reference;
  
    // Number of iterations
    private int numIterations = 5;
    
    // A flag to indicate whether or not the filter is being used for the first time (i.e. for training)
    // or for subsequent times (i.e for testing)
    private boolean isTraining=true;
    
    // Flag to indicate whether or not scaling is allowed (because sometimes size differences are important)
    private boolean allowScaling=true;
    
    // Return info about the filter
    public String globalInfo() {
        return "Generalized Procrustes Analysis for 2D points filter.";
    }
      
    // Getters and setters 
    public int getSeed() { return seed; }
    public void setSeed(int val) { seed=val; }
    public int getNumIterations() { return numIterations; }
    public void setNumIterations(int val) { numIterations=val; }
    public boolean getAllowScaling() { return allowScaling; }
    public void setAllowScaling(boolean val) { allowScaling=val; }
    
    // The filter doesn't change the format of the data, therefore simply returns the inputFormat
    protected Instances determineOutputFormat(Instances inputFormat) {
        return inputFormat;
    }
   
    // The filter applies to any dataset and does not require a class attribute
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.enableAllAttributes();
        result.enableAllClasses();
        result.enable(Capabilities.Capability.NO_CLASS);  
        return result;
    }
  
    // Setting the options
    public void setOptions(String[] options) throws Exception{
        String optionString;
        // Set the random number seed
        optionString = Utils.getOption('S',options);
        if (optionString.length()!=0) {
            setSeed( Integer.parseInt(optionString) );
        }
        // Set the number of iterations
        optionString = Utils.getOption('I',options);
        if (optionString.length()!=0) {
            setNumIterations( Integer.parseInt(optionString) );
        }
        // Set whether or not to allow scaling
        optionString = Utils.getOption('C',options);
        if (optionString.length()!=0) {
            setAllowScaling( Boolean.parseBoolean(optionString) );
        }
    }
    
    // Getting the options
    public String[] getOptions() {
        String[] options=new String[6];
        int current=0;
        // Get random number seed
        options[current++] = "-S";
        options[current++] = ""+getSeed();
        // Get the number of iterations
        options[current++] = "-I";
        options[current++] = ""+getNumIterations();
        // Get whether to allow scaling or not
        options[current++] = "-C";
        options[current++] = ""+getAllowScaling();
        while(current<options.length){
            options[current++]="";
        }
        return options;
    }
    
    // Enumeration describing all the options
    public Enumeration listOptions() {
        Vector options = new Vector(3);
        options.addElement(new Option("Random number seed for selecting the initial reference","S",42,"-S"));
        options.addElement(new Option("Number of iterations with which to update the reference","I",5,"-I"));
        options.addElement(new Option("Whether to allow scaling or not","C",1,"-C"));
        return options.elements();
    }
    
    // Tool tip texts
    public String numIterationsTipText() {
        return "Number of iterations with which to update the reference";
    }
    public String seedTipText() {
        return "Random number seed for selecting the initial reference";
    }
    public String allowScalingTipText() {
        return "Whether or not to allow scaling of shapes";
    }
    
    // This method is the main workhorse for the filter, and takes a dataset and processes it
    public Instances process(Instances data) {
        // If there is no data, simply do nothing
        if (data.numInstances()==0) return data;
        // If this is the first time the filter has run, some setup needs to be done
        if (numPoints==-1) {
            // Save the header so that attribute names can be retrieved later
            trainingDataHeader = new Instances(data,0);
            // Pre calculate the number of 2D points per example
            numPoints = numPoints();
            // Create the random number generator
            rng=new Random(seed);
            // Select a random reference implementation
            reference=data.instance( rng.nextInt( data.numInstances() ) );
        }
        // Do the training work
        if (isTraining) {
            for (int iterationIndex=0; iterationIndex<numIterations; iterationIndex++) {
                // Translate, scale and rotate all of the data
                for (int instanceIndex=0; instanceIndex<data.numInstances(); instanceIndex++) {
                    translate( data.instance(instanceIndex) );
                    if (allowScaling) scale( data.instance(instanceIndex) );
                    rotate( data.instance(instanceIndex) );
                }
                // Set the new reference to the mean of the dataset for the next iteration  
                reference=mean(data);

            }
            isTraining=false; // Done training
        } else {
            // Test phase. There is likely only one example. We translate, scale and rotate the examples but
            // we don't iterate and we don't update the reference
            for (int instanceIndex=0; instanceIndex<data.numInstances(); instanceIndex++) {
                translate( data.instance(instanceIndex) );
                if (allowScaling) scale( data.instance(instanceIndex) );
                rotate( data.instance(instanceIndex) );
            }
        }
        // Done
        return data;
    }
  
    
    // Helper method to calculate the mean of a dataset
    private Instance mean(Instances data) {
        Instance result=new DenseInstance(data.numAttributes());
        for (int attIndex=0; attIndex<data.numAttributes(); attIndex++) {
            result.setValue(attIndex,0);
        }
        for (int instanceIndex=0; instanceIndex<data.numInstances(); instanceIndex++) 
        for (int attIndex=0; attIndex<data.numAttributes(); attIndex++) {  
            result.setValue(attIndex, result.value(attIndex)+data.instance(instanceIndex).value(attIndex));
        }
        for (int attIndex=0; attIndex<data.numAttributes(); attIndex++) {
            result.setValue( attIndex, result.value(attIndex)/data.numInstances() );
        }
        return result;
    }
    
    // Helper method to calculate mean RSE error across an entire dataset
    private double mrse(Instances data) {
        float error=0;
        for (int instanceIndex=0; instanceIndex<data.numInstances(); instanceIndex++) {
            error += rse(data.instance(instanceIndex));
        }
        return error;
    }
    
    // Helper method to calculate the RSE error (i.e. the procrustes distance) between an example and the reference
    private double rse(Instance instance) {
        double error=0;
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt    = getPoint(instance, pointIndex);
            double[] refpt = getPoint(reference, pointIndex);
            double diffx = pt[0]-refpt[0];
            double diffy = pt[1]-refpt[1];
            error += (diffx * diffx) + (diffy*diffy);
        }
        error = Math.sqrt(error);
        return error;
    }
    
    // Helper method to rotate a single example to minimize error with the reference
    private void rotate(Instance instance) {
        // Compute the angle between the example and the reference
        double num=0, dem=0;
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt    = getPoint(instance, pointIndex);
            double[] refpt = getPoint(reference, pointIndex);
            num += pt[0]*refpt[1]-pt[1]*refpt[0];
            dem += pt[0]*refpt[0]+pt[1]*refpt[1];
        }
        double theta=Math.atan2(num,dem);
        // Rotate the example -- this *should* reduce the error :)
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt    = getPoint(instance, pointIndex);
            double[] newpt = new double[2];
            newpt[0] = Math.cos(theta)*pt[0] - Math.sin(theta)*pt[1];
            newpt[1] = Math.sin(theta)*pt[0] + Math.cos(theta)*pt[1];
            setPoint(instance, pointIndex, newpt);
        }
    }
    
    
    // Helper method to scale a single example average unit length
    // Assumes that the mean is already (0,0), i.e. the example has already been translated
    private void scale(Instance instance) {
        double ssd=0;
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt = getPoint(instance, pointIndex);
            ssd+= pt[0]*pt[0]+pt[1]*pt[1];
        }
        ssd = Math.sqrt(ssd/2);
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt = getPoint(instance, pointIndex);
            setPoint(instance, pointIndex, new double[]{ pt[0]/ssd, pt[1]/ssd });
        }    
    }
    
    // Helper method to scale a single example average unit length
    private void __scale(Instance instance) {
        // Calculate the mean point magnitude
        double meanMagnitude=0;
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt = getPoint(instance, pointIndex);
            meanMagnitude+= Math.sqrt(pt[0]*pt[0]+pt[1]*pt[1]);
        }
        // Average the magnitude
        meanMagnitude /= numPoints;
        // scale the points by the magnitude to the average magnitude is unit
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt = getPoint(instance, pointIndex);
            setPoint(instance, pointIndex, new double[]{ pt[0]/meanMagnitude, pt[1]/meanMagnitude });
        }
        // Done
    }
    
    // Helper method to translate a single example to the origin
    private void translate(Instance instance) {
        // Sum up the points on the example
        double meanX=0, meanY=0;
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt = getPoint(instance, pointIndex);
            meanX+=pt[0];
            meanY+=pt[1];
        }
        // Average them
        meanX /= numPoints;
        meanY /= numPoints;
        // Subtract the mean
        for (int pointIndex=1; pointIndex<=numPoints; pointIndex++) {
            double[] pt = getPoint(instance, pointIndex);
            setPoint(instance, pointIndex, new double[]{ pt[0]-meanX, pt[1]-meanY });
        }
        // Done
    }
    
    
    // Helper method to determine how many 2D points are encoded per instance
    // Expects attribute names to be in format "x1,y1,x2,y2,..."
    // Expects the trainingDataHeader to be set 
    private int numPoints() {
        Attribute x,y;
        int count=0;
        while(true){
            x = trainingDataHeader.attribute("x"+(count+1));
            y = trainingDataHeader.attribute("y"+(count+1));
            if (x==null || y==null) break;
            count++;
        }
        return count;
    }
  
    // Helper method to get one point from an instance. 
    // The point index should be between 1 and numPoints.
    private double[] getPoint(Instance instance, int index) {
        Attribute att1 = trainingDataHeader.attribute("x"+index);
        Attribute att2 = trainingDataHeader.attribute("y"+index);
        return new double[] {instance.value(att1),instance.value(att2)};
    }
  
    // Helper method to set one point on an instance.
    // The point index should be between 1 and numPoints.
    // Double array pt should be of size 2.
    private void setPoint(Instance instance, int index, double[] pt) {
        Attribute att1 = trainingDataHeader.attribute("x"+index);
        Attribute att2 = trainingDataHeader.attribute("y"+index);
        instance.setValue(att1, pt[0]);
        instance.setValue(att2, pt[1]);
    }
  
   
  

  
}