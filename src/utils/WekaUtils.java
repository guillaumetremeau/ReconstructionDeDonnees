package utils;
 
import java.io.IOException;
import weka.classifiers.Classifier; 
import weka.classifiers.functions.Logistic; 
import weka.classifiers.functions.MultilayerPerceptron; 
import weka.classifiers.functions.SMO; 
import weka.classifiers.trees.DecisionStump; 
import weka.classifiers.trees.J48; 
import weka.classifiers.trees.RandomForest; 
import weka.core.Attribute; 
import weka.core.OptionHandler; 
 
import java.text.MessageFormat; 
import java.util.HashMap;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList; 
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.functions.supportVector.RegSMOImproved;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
 
public class WekaUtils { 
 
   
    
    public  static Map EvaluateRF(String fileName, int[][] img, int[][] imgBruitee, int percent, int nbpiece, int idpiece) throws IOException, Exception {
        double[][] regressionData = Utils.createVariables(imgBruitee, Utils.radius1, Utils.radius2, Utils.radius3,Utils.minElementR1, Utils.minElementR2, Utils.minElementR2);
            
        return  EvaluateRF(regressionData,  img) ;      
      
    }
    
    public  static Map Evaluate( int[][] img, int[][] imgBruitee,  String regFunction) throws IOException, Exception {
        double[][] regressionData = Utils.createVariables(imgBruitee, Utils.radius1, Utils.radius2, Utils.radius3,Utils.minElementR1, Utils.minElementR2, Utils.minElementR2);
        
        switch(regFunction){
            //case "RF": return EvaluateRF(regressionData,  img) ; 
            
            case "SVM": return EvaluateSVMReg(regressionData,  img) ; 
            case "RT": return EvaluateRT(regressionData,  img) ;
            case "MR": return EvaluateMR(regressionData,  img) ;
            case "CART": return  EvaluateCart(regressionData,  img) ;            
            default :  return  EvaluateRF(regressionData,  img) ;      
        
        }      
      
    }
    
    
    
     public static double[] getOneRegressionLine(double[][] regressionData, int x, int y){        
            
            for(double []regLine:regressionData){
            
                if(regLine[0]==x && regLine[1]==y) 
                {
                    double tab[]=new double[16];
                     System.arraycopy(regLine, 0, tab, 0, regLine.length);
                     //tab[0]=x;
                    // tab[1]=y;
                    return tab;
                }
            }
            
            return null;
        }
    
       public  static Map Evaluate2(double[][] regressionData, int[][] img, int[]pos,String regFunction,int percent, int nbpiece, int idpiece) throws IOException, Exception {

           double[][] regressionData2= new double[img.length*img[0].length][16];
      
          // System.out.println("X="+pos[0]+", Y="+pos[1]);
                   int z=0;
           for(int i=0;i<img.length;i++){
            for(int j=0;j<img[0].length;j++){ 
                
                int x=pos[0]+i+1,y=pos[1]+j+1;
               
                double line[]=getOneRegressionLine(regressionData,x,y);
                 line[0]=i+1;
                 line[1]=j+1;
              regressionData2[z++]=line;
                
               // regressionData2[y][0]=i+1;
               //regressionData2[y][1]=j+1;
            //   regressionData2[y][3]=regressionData[];
            }
        
            
          
            
        }
              
        //CSVUtils.writecsv(regressionData2, "back_white-"+ percent+"-"+nbpiece+"-"+idpiece+".csv");
       /// System.out.println("Nestor"+y);
        //System.exit(0);
       switch(regFunction){
            //case "RF": return EvaluateRF(regressionData,  img) ; 
            
            case "SVM": return EvaluateSVMReg(regressionData2,  img) ; 
            case "RT": return EvaluateRT(regressionData2,  img) ;
            case "MR": return EvaluateMR(regressionData2,  img) ;
            case "CART": return  EvaluateCart(regressionData2,  img) ;            
            default :  return  EvaluateRF(regressionData2,  img) ;      
        
        }     
      
      
    }
       public  static Map Evaluate2(double[][] regressionData, int[][] img, String regFunction) throws IOException, Exception {

           double[][] regressionData2= new double[img.length*img[0].length][16];
        int y=0;
        for(int i=0;i<img.length;i++){
            for(int j=0;j<img[0].length;j++){                
               
              regressionData2[y++]=getOneRegressionLine(regressionData,i+1,j+1);
                
               // regressionData2[y][0]=i+1;
               //regressionData2[y][1]=j+1;
            //   regressionData2[y][3]=regressionData[];
            }
        
            
          
            
        }
              
       // CSVUtils.writecsv(regressionData2, "back_white-"+ percent+"-"+nbpiece+"-"+idpiece+".csv");
       
       /// System.out.println("Nestor"+y);
        //System.exit(0);
       switch(regFunction){
            //case "RF": return EvaluateRF(regressionData,  img) ; 
            
            case "SVM": return EvaluateSVMReg(regressionData2,  img) ; 
            case "RT": return EvaluateRT(regressionData2,  img) ;
            case "MR": return EvaluateMR(regressionData2,  img) ;
            case "CART": return  EvaluateCart(regressionData2,  img) ;            
            default :  return  EvaluateRF(regressionData2,  img) ;      
        
        }     
      
      
    }

    public static Map EvaluateMR(double[][] regressionData, int[][] originalimage) throws Exception{
     
          double[][] trainingData = Utils.getTrainingData(regressionData);
          
          double[][] testData = Utils.getTestData(regressionData);
         
          //double[][] testData = Utils.getTestData(variables);
        int i = 0;
        for (double[] tuple : testData) {
            int x = (int) tuple[0];
            int y = (int) tuple[1];
            testData[i++][2] = originalimage[x - 1][y - 1];

        }
        Instances trainingInstances=MyUtilsForWekaInstanceHelper.getInstances(trainingData);        
        Instances testInstances=MyUtilsForWekaInstanceHelper.getInstances(testData);
        
        
         int [] indices={2,3,4,5,6,7,8,9,10,11,12,13,14,15};

         
           Remove attributeFilter1 = new Remove();
           attributeFilter1.setInvertSelection(true);
           attributeFilter1.setAttributeIndicesArray(indices);
           attributeFilter1.setInputFormat(trainingInstances);
           trainingInstances = Filter.useFilter(trainingInstances, attributeFilter1);
           
           
           Remove attributeFilter2 = new Remove();
           attributeFilter2.setInvertSelection(true);
           attributeFilter2.setAttributeIndicesArray(indices);
           attributeFilter2.setInputFormat(testInstances);
           testInstances = Filter.useFilter(testInstances, attributeFilter2);
        
         
        LinearRegression classifier = new LinearRegression();
        
        classifier.buildClassifier(trainingInstances);
        Evaluation eval = new Evaluation(trainingInstances);
        eval.evaluateModel(classifier, testInstances);
        
        ArrayList<Prediction> Result = eval.predictions();
       // int i = 0;
       int predictedImage[][]= new int[originalimage.length][originalimage[0].length];
       for(double[] pixel:trainingData){
            int x = (int) pixel[0], y = (int) pixel[1];
            predictedImage[x-1][y-1]=(int)pixel[2];
       }
       i=0;
        for (Prediction predicted : Result) {
            int x = (int) testData[i][0], y = (int) testData[i++][1];          
            predictedImage[x - 1][y - 1] = (int) predicted.predicted();
        }
        
        Map map = new HashMap<>();
        
        map.put("eval", eval);
        map.put("img", predictedImage);
        return map;
     }  
    
    public static Map EvaluateRF(double[][] regressionData, int[][] originalimage) throws Exception{
     
          double[][] trainingData = Utils.getTrainingData(regressionData);
          
          double[][] testData = Utils.getTestData(regressionData);
         
          //double[][] testData = Utils.getTestData(variables);
        int i = 0;
        for (double[] tuple : testData) {
            int x = (int) tuple[0];
            int y = (int) tuple[1];
            testData[i++][2] = originalimage[x - 1][y - 1];

        }
          
        
        
        Instances trainingInstances=MyUtilsForWekaInstanceHelper.getInstances(trainingData);        
        Instances testInstances=MyUtilsForWekaInstanceHelper.getInstances(testData);
         
     
       /* int [] indices={0,1,2,3,4,5,6,7,8,9,10,11,12,12,13,14,15};

         
           Remove attributeFilter1 = new Remove();
           attributeFilter1.setInvertSelection(true);
           attributeFilter1.setAttributeIndicesArray(indices);
           attributeFilter1.setInputFormat(trainingInstances);
           trainingInstances = Filter.useFilter(trainingInstances, attributeFilter1);
           
           
           Remove attributeFilter2 = new Remove();
           attributeFilter2.setInvertSelection(true);
           attributeFilter2.setAttributeIndicesArray(indices);
           attributeFilter2.setInputFormat(testInstances);
           testInstances = Filter.useFilter(testInstances, attributeFilter2);
           */

        final Classifier classifier = new RandomForest();
        
        classifier.buildClassifier(trainingInstances);
        Evaluation eval = new Evaluation(trainingInstances);
        eval.evaluateModel(classifier, testInstances);

        
        // System.out.println("----Training--"+trainingData.length+" ; Test---"+testData.length+"    img"+originalimage.length*originalimage[0].length);
        
        ArrayList<Prediction> Result = eval.predictions();
       // int i = 0;
       int predictedImage[][]= new int[originalimage.length][originalimage[0].length];
       for(double[] pixel:trainingData){
       if(pixel!=null){   int x = (int) pixel[0], y = (int) pixel[1];
            
            
            predictedImage[x-1][y-1]=(int)pixel[2];
       }
       }
       i=0;

//     System.out.println("----"+Result.size());
      
     if(Result!=null)
     for (Prediction predicted : Result) {
          if(predicted !=null){  
          
          int x = (int) testData[i][0], y = (int) testData[i++][1];          
            predictedImage[x - 1][y - 1] = (int) predicted.predicted();
          }
        }
        
        Map map = new HashMap<>();
        
        map.put("eval", eval);
        map.put("img", predictedImage);
        return map;
     }  
    
     
     public static Map EvaluateRT(double[][] regressionData, int[][] originalimage) throws Exception{
     
          double[][] trainingData = Utils.getTrainingData(regressionData);
          
          double[][] testData = Utils.getTestData(regressionData);
         
          //double[][] testData = Utils.getTestData(variables);
        int i = 0;
        for (double[] tuple : testData) {
            int x = (int) tuple[0];
            int y = (int) tuple[1];
            testData[i++][2] = originalimage[x - 1][y - 1];

        }
        Instances trainingInstances=MyUtilsForWekaInstanceHelper.getInstances(trainingData);        
        Instances testInstances=MyUtilsForWekaInstanceHelper.getInstances(testData);
         
        
       RandomTree tree= new RandomTree();
       tree.setSeed(5);
       tree.buildClassifier(trainingInstances);
       
        
        final Classifier classifier = new RandomTree();
        
        classifier.buildClassifier(trainingInstances);
        Evaluation eval = new Evaluation(trainingInstances);
        eval.evaluateModel( tree, testInstances);
        
        ArrayList<Prediction> Result = eval.predictions();
       // int i = 0;
       int predictedImage[][]= new int[originalimage.length][originalimage[0].length];
       for(double[] pixel:trainingData){
            int x = (int) pixel[0], y = (int) pixel[1];
            predictedImage[x-1][y-1]=(int)pixel[2];
       }
       i=0;
        for (Prediction predicted : Result) {
            int x = (int) testData[i][0], y = (int) testData[i++][1];          
            predictedImage[x - 1][y - 1] = (int) predicted.predicted();
        }
        
        Map map = new HashMap<>();
        
        map.put("eval", eval);
        map.put("img", predictedImage);
        return map;
     }  
    
     
     public static Map EvaluateCart(double[][] regressionData, int[][] originalimage) throws Exception{
     
          double[][] trainingData = Utils.getTrainingData(regressionData);
          
          double[][] testData = Utils.getTestData(regressionData);
         
          //double[][] testData = Utils.getTestData(variables);
        int i = 0;
        for (double[] tuple : testData) {
            int x = (int) tuple[0];
            int y = (int) tuple[1];
            testData[i++][2] = originalimage[x - 1][y - 1];

        }
        Instances trainingInstances=MyUtilsForWekaInstanceHelper.getInstances(trainingData);        
        Instances testInstances=MyUtilsForWekaInstanceHelper.getInstances(testData);
         
        
         int [] indices={2,3,4,5,6,7,8,9,10,11,12,13,14,15};

         
           Remove attributeFilter1 = new Remove();
           attributeFilter1.setInvertSelection(true);
           attributeFilter1.setAttributeIndicesArray(indices);
           attributeFilter1.setInputFormat(trainingInstances);
           trainingInstances = Filter.useFilter(trainingInstances, attributeFilter1);
           
           
           Remove attributeFilter2 = new Remove();
           attributeFilter2.setInvertSelection(true);
           attributeFilter2.setAttributeIndicesArray(indices);
           attributeFilter2.setInputFormat(testInstances);
           testInstances = Filter.useFilter(testInstances, attributeFilter2);
        
        
       REPTree tree= new REPTree();
       tree.setSeed(5);
       tree.buildClassifier(trainingInstances);
       
        
        final Classifier classifier = new RandomTree();
        
        classifier.buildClassifier(trainingInstances);
        Evaluation eval = new Evaluation(trainingInstances);
        eval.evaluateModel( tree, testInstances);
        
        ArrayList<Prediction> Result = eval.predictions();
       // int i = 0;
       int predictedImage[][]= new int[originalimage.length][originalimage[0].length];
       for(double[] pixel:trainingData){
            int x = (int) pixel[0], y = (int) pixel[1];
            predictedImage[x-1][y-1]=(int)pixel[2];
       }
       i=0;
        for (Prediction predicted : Result) {
            int x = (int) testData[i][0], y = (int) testData[i++][1];          
            predictedImage[x - 1][y - 1] = (int) predicted.predicted();
        }
        
        Map map = new HashMap<>();
        
        map.put("eval", eval);
        map.put("img", predictedImage);
        return map;
     }  
    
     
     public static Map EvaluateSVMReg(double[][] regressionData, int[][] originalimage) throws Exception{
     
          double[][] trainingData = Utils.getTrainingData(regressionData);
          
          double[][] testData = Utils.getTestData(regressionData);
         
          //double[][] testData = Utils.getTestData(variables);
        int i = 0;
        for (double[] tuple : testData) {
            int x = (int) tuple[0];
            int y = (int) tuple[1];
            testData[i++][2] = originalimage[x - 1][y - 1];

        }
        Instances trainingInstances=MyUtilsForWekaInstanceHelper.getInstances(trainingData);        
        Instances testInstances=MyUtilsForWekaInstanceHelper.getInstances(testData);
         
        
         int [] indices={2,3,4,5,6,7,8,9,10,11,12,13,14,15};

         
           Remove attributeFilter1 = new Remove();
           attributeFilter1.setInvertSelection(true);
           attributeFilter1.setAttributeIndicesArray(indices);
           attributeFilter1.setInputFormat(trainingInstances);
           trainingInstances = Filter.useFilter(trainingInstances, attributeFilter1);
           
           
           Remove attributeFilter2 = new Remove();
           attributeFilter2.setInvertSelection(true);
           attributeFilter2.setAttributeIndicesArray(indices);
           attributeFilter2.setInputFormat(testInstances);
           testInstances = Filter.useFilter(testInstances, attributeFilter2);
        
         SMOreg svm= new SMOreg();
       
            double cValue = 1;
	    double gammaValue = -5;
	    Kernel kernelValue = new RBFKernel();
	    double c = 1;Math.pow(2, cValue);
            
            RegSMOImproved opti= new  RegSMOImproved();
                opti.setEpsilon(1.0E-12);
                opti.setEpsilonParameter(0.001);
                opti.setTolerance(0.001);
                opti.setSeed(1);
                opti.setUseVariant1(true);
                
            svm.setRegOptimizer(opti);
         
	    double gamma = 0.1;//Math.pow(2, gammaValue);
		//
	   // SMO smo = new SMO();
           ((RBFKernel) kernelValue).setGamma(gamma);
          // svm.setFilterType(SelectedTag);
           
            svm.setKernel(kernelValue);
            svm.setC(c);
           
       
        
         svm.buildClassifier(trainingInstances);
        Evaluation eval = new Evaluation(trainingInstances);
        eval.evaluateModel( svm, testInstances);
        
        ArrayList<Prediction> Result = eval.predictions();
       // int i = 0;
       int predictedImage[][]= new int[originalimage.length][originalimage[0].length];
       for(double[] pixel:trainingData){
            int x = (int) pixel[0], y = (int) pixel[1];
            predictedImage[x-1][y-1]=(int)pixel[2];
       }
       i=0;
        for (Prediction predicted : Result) {
            int x = (int) testData[i][0], y = (int) testData[i++][1];          
            predictedImage[x - 1][y - 1] = (int) predicted.predicted();
        }
        
        Map map = new HashMap<>();
        
        map.put("eval", eval);
        map.put("img", predictedImage);
        return map;
     }  
    
     
     
         
     
    
    public static final String CLASSES_ATTR_NAME = "classes"; 
    public static final String FEATURE_PREFIX = "feature-"; 
 
    public static Classifier makeClassifier(String wekaClassifier, String[] options) throws Exception { 
        switch (WekaClassificationAlgorithms.valueOf(wekaClassifier)) { 
            case decisionTree: 
                J48 j48 = new J48(); 
                setOptionsForWekaPredictor(options, j48); 
                return j48; 
            case svm: 
                SMO smo = new SMO(); 
                smo.setNumFolds(-1); 
                setOptionsForWekaPredictor(options, smo); 
                return smo; 
          
            case randomForest: 
                RandomForest forest = new RandomForest(); 
                setOptionsForWekaPredictor(options, forest); 
                return forest; 
            case decisionStump: 
                DecisionStump stump = new DecisionStump(); 
                setOptionsForWekaPredictor(options, stump); 
                return stump; 
            case perceptron: 
                MultilayerPerceptron perceptron = new MultilayerPerceptron(); 
                setOptionsForWekaPredictor(options, perceptron); 
                return perceptron; 
            default: 
                return new SMO(); 
        } 
    } 
 
  
    public static ArrayList<Attribute> makeFeatureVectorForBatchClustering(int noOfAttributes, int numClasses) { 
        // Declare FAST VECTOR 
        ArrayList<Attribute> attributeInfo = new ArrayList<Attribute>(); 
 
        // Declare FEATURES and add them to FEATURE VECTOR 
        for (int i = 0; i < noOfAttributes; i++) { 
            attributeInfo.add(new Attribute(MessageFormat.format("feature-{0}", i))); 
        } 
 
        System.err.println("DEBUG: no. of attributes = " + attributeInfo.size()); 
        return attributeInfo; 
    } 
 
    public static ArrayList<Attribute> makeFeatureVectorForBinaryClassification(int noOfAttributes) { 
        ArrayList<Attribute> attributeInfo = new ArrayList<Attribute>(); 
        // Declare FEATURES and add them to FEATURE VECTOR 
        for (int i = 0; i < noOfAttributes; i++) { 
            attributeInfo.add(new Attribute(MessageFormat.format("feature-{0}", i))); 
        } 
        // last element in a FEATURE VECTOR is the category 
        ArrayList<String> classNames = new ArrayList<String>(2); 
        for (int i = 1; i <= 2; i++) { 
            classNames.add(MessageFormat.format("class-{0}", String.valueOf(i))); 
        } 
        Attribute classes = new Attribute(CLASSES_ATTR_NAME, classNames); 
        // last element in a FEATURE VECTOR is the category 
        attributeInfo.add(classes); 
        System.err.println("DEBUG: no. of attributes = " + attributeInfo.size()); 
        return attributeInfo; 
    } 
 
    public static ArrayList<Attribute> makeFeatureVectorForOnlineClustering(int noOfClusters, int noOfAttributes) { 
        // Declare FAST VECTOR 
        ArrayList<Attribute> attributeInfo = new ArrayList<Attribute>(); 
 
        // Declare FEATURES and add them to FEATURE VECTOR 
        for (int i = 0; i < noOfAttributes; i++) { 
            attributeInfo.add(new Attribute(MessageFormat.format("feature-{0}", i))); 
        } 
 
        System.err.println("DEBUG: no. of attributes = " + attributeInfo.size()); 
        return attributeInfo; 
    } 
 
  
    public static void setOptionsForWekaPredictor(String[] options, OptionHandler randomForest) throws Exception { 
        if (options != null) { 
            randomForest.setOptions(options); 
        } 
    } 
}