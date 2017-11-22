package Inpainting;

import java.awt.image.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import utils.CSVUtils;
import utils.MyUtilsForWekaInstanceHelper;
import utils.Utils;
import utils.WekaUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

/**
 *
 * @author koueya
 */
public class ImageInpaint {

    String dataPath = "data";
    String imagePath = "graphics";
    private final int radius1 = 10;
    private final int radius2 = 20;
    private final int radius3 = 35 ;
    private final int minElementR1 = 2, minElementR2 = 5, minElementR3 = 10;
    private final double[] percents = {0.3,0.4,0.5,0.6,0.7};
    private final int[] pieces = {4,9,16};


public ImageInpaint(String PathName,String regFunction) {

        File path = new File(PathName);
        ArrayList fileList = Utils.listerRepertoire(path);

        IntStream.range(0, fileList.size()).parallel().forEach(i -> {
            try {
               
                 switch(regFunction){
                     case "SVM": this.SVM2((String) fileList.get(i), PathName); break;
                     case "MR": this.LinearRegression((String) fileList.get(i), PathName); break;
                     case "CART": this.CART2((String) fileList.get(i), PathName); break;
                     case "RT": this.RT((String) fileList.get(i), PathName); break;                     
                     default: this.RF2((String) fileList.get(i), PathName);
              }
               // testwithRvaluesMoceau((String) fileList.get(i), PathName,4);
            } catch (Exception exe) {
                Logger.getLogger(ImageInpaint.class.getName()).log(Level.SEVERE, null, exe);
            }

        });

    }

public ImageInpaint(String PathName,String regFunction, boolean border) {

        File path = new File(PathName);
        ArrayList fileList = Utils.listerRepertoire(path);

        IntStream.range(0, fileList.size()).parallel().forEach(i -> {
            try {
                if(border){
                 switch(regFunction){
                     case "SVM": this.SVM2((String) fileList.get(i), PathName); break;
                     case "MR": this.LinearRegression((String) fileList.get(i), PathName); break;
                     case "CART": this.CART2((String) fileList.get(i), PathName); break;
                     case "RT": this.RT((String) fileList.get(i), PathName); break;                     
                     default: this.RF2((String) fileList.get(i), PathName);
                     
              }
                 }
                else {
                         
                       switch(regFunction){
                     case "SVM": this.SVM((String) fileList.get(i), PathName); break;
                     case "MR": this.LinearRegression((String) fileList.get(i), PathName); break;
                     case "CART": this.CART((String) fileList.get(i), PathName); break;
                     case "RT": this.RT((String) fileList.get(i), PathName); break;                     
                     default: this.RF((String) fileList.get(i), PathName);
                     
              }   
                         }
               // testwithRvaluesMoceau((String) fileList.get(i), PathName,4);
            } catch (Exception exe) {
                Logger.getLogger(ImageInpaint.class.getName()).log(Level.SEVERE, null, exe);
            }

        });

    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    public boolean testwithRvalues(String fileName, String path) throws IOException, Exception {

        boolean response = true;
        IImage image = new IImage(fileName, path);
        String debug = " [" + fileName + "  ";
        for (double percent : percents) {
            debug += " | percent=" + percent + "  ";

            IImage imagecopie = new IImage(image.img);

            ArrayList<double[]> positionbruit = this.getBruitFromCsv("colisium.csv");

            imagecopie.GenerateBruit(positionbruit);
            int percent2 = (int) (percent * 100);

            double[][] variables = Utils.createVariables(imagecopie.getPixelMatrix(), radius1, radius2, radius3, minElementR1, minElementR2, minElementR3);
            for (int i = 0; i < variables.length; i++) {
                for (int j = 0; j < variables[0].length; j++) {
                    if (variables[i][j] != positionbruit.get(i)[j]) {

                        response = false;

                    }
                }
            }

        }
        return response;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    public boolean testwithRvaluesMoceau(String fileName, String path, int nbpiece) throws IOException, Exception {

        boolean response = true;
        IImage image = new IImage(fileName, path);
        String debug = " [" + fileName + "  ";
        for (double percent : percents) {
            debug += " | percent=" + percent + "  ";

            IImage imagecopie = new IImage(image.img);
            ArrayList<IImage> BimageList = imagecopie.sliceImage(nbpiece);
            int y=1;
            for (IImage smalimage : BimageList) 
           // IImage smalimage = BimageList.get(3);
            {

                ArrayList<double[]> positionbruit = this.getBruitFromCsv("coliseum_30_morceau_"+y+".csv");
                
               // System.out.println(positionbruit.size()+" -- ");

                smalimage.GenerateBruit(positionbruit);
                int percent2 = (int) (percent * 100);

                
              //  Map map2 = ExecuteValue(fileName, (int[][]) smalimage.getPixelMatrix(), smalimage.getPixelMatrix(), percent2, 4, y);
                //System.exit(0);

                double[][] variables = Utils.createVariables(smalimage.getPixelMatrix(), radius1, radius2, radius3, minElementR1, minElementR2, minElementR3);
                for (int i = 0; i < variables.length; i++) {
                    for (int j = 0; j < variables[0].length; j++) {
                        if (variables[i][j] != positionbruit.get(i)[j]) {
                            
                            System.out.println(variables[i][0]+"--"+variables[i][1]+" -----j="+j+"  pxel="+variables[i][2]+" "+variables[i][j]+"----"+positionbruit.get(i)[j]);
                            //System.exit(0);
                            //response = false;

                        }
                    }
                }
                y++;
            }
        }
        return response;
    }

 
    ////------------------------Test  ------------------------------------------------------ 
    public void RF(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " RF;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            Map map =   WekaUtils.Evaluate( imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(), "RF");
            Evaluation eval = (Evaluation) map.get("eval");

            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ;piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);

                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece);

                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate( (int[][]) OimageList.get(i).getPixelMatrix(), (int[][]) BimageList.get(i).getPixelMatrix(),"RF");

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
               // debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_rf_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }

    
       ////------------------------Test  ------------------------------------------------------ 
    public void RF2(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " RF2;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            double[][] regressionData = Utils.createVariables(imagecopie.getPixelMatrix(), Utils.radius1, Utils.radius2, Utils.radius3,Utils.minElementR1, Utils.minElementR2, Utils.minElementR2);

            Map map =   WekaUtils.Evaluate( imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(), "RF");
            Evaluation eval = (Evaluation) map.get("eval");

  
            // CSVUtils.writecsv(regressionData, "back_white-"+ percent+".csv");

            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ;piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);
                Map position = new HashMap<>();
                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece,position);

                System.out.println(position.size()+"--Size-");
               // System.exit(0);
                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate2(regressionData,(int[][]) OimageList.get(i).getPixelMatrix(),(int[])position.get(i),"RF",30,piece,i+1);

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
               // debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_rf2_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }

    
    
    
          ////------------------------Test  ------------------------------------------------------ 
    public void SVM2(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " SVM2;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            double[][] regressionData = Utils.createVariables(imagecopie.getPixelMatrix(), Utils.radius1, Utils.radius2, Utils.radius3,Utils.minElementR1, Utils.minElementR2, Utils.minElementR2);

            Map map =   WekaUtils.Evaluate( imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(), "SVM");
            Evaluation eval = (Evaluation) map.get("eval");

  
            
            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ;piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);

                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece);

                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate2( regressionData,(int[][]) OimageList.get(i).getPixelMatrix(),"SVM");

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
               // debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_rf2_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }

    
    
        public void CART2(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " CART2;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            double[][] regressionData = Utils.createVariables(imagecopie.getPixelMatrix(), Utils.radius1, Utils.radius2, Utils.radius3,Utils.minElementR1, Utils.minElementR2, Utils.minElementR2);

            Map map =   WekaUtils.Evaluate( imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(), "CART");
            Evaluation eval = (Evaluation) map.get("eval");

  
            
            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ;piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);

                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece);

                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate2( regressionData,(int[][]) OimageList.get(i).getPixelMatrix(),"CART");

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
               // debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_cart2_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }

   
    
    ////------------------------Test  ------------------------------------------------------ 
    public void SVM(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " SVM;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            Map map =   WekaUtils.Evaluate(imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(),"SVM");
            Evaluation eval = (Evaluation) map.get("eval");

            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ;piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);

                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece);

                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate((int[][]) OimageList.get(i).getPixelMatrix(), (int[][]) BimageList.get(i).getPixelMatrix(),"SVM");

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
                //debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_svm_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }
    
    
    ////------------------------Test  ------------------------------------------------------ 
    public void LinearRegression(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " MR ;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            Map map =   WekaUtils.Evaluate(imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(),"MR");
            Evaluation eval = (Evaluation) map.get("eval");

            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ;piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);

                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece);

                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate( (int[][]) OimageList.get(i).getPixelMatrix(), (int[][]) BimageList.get(i).getPixelMatrix(),"SVM");

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
               //debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_LR_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }
    public void RT(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " RT;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            Map map =   WekaUtils.Evaluate( imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(),"RT");
            Evaluation eval = (Evaluation) map.get("eval");

            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += ";piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);

                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece);

                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate((int[][]) OimageList.get(i).getPixelMatrix(), (int[][]) BimageList.get(i).getPixelMatrix(),"RT");

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
               // debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_RT_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }
    

 public void CART(String fileName, String path) throws IOException, Exception {

        IImage image = new IImage(fileName, path);
        String debug = " CART;" + fileName + "  ";
        for (double percent : percents) {
            debug += " ;" + percent + " ; ";

            IImage imagecopie = new IImage(image.img);           
            
             imagecopie.GenerateBruit(percent);
             
            int percent2 = (int) (percent * 100);

            imagecopie.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            Map map =   WekaUtils.Evaluate( imagecopie.getOriginalImageCopie().getPixelMatrix(), imagecopie.getPixelMatrix(), "CART");
            Evaluation eval = (Evaluation) map.get("eval");

            debug += " MAE_global;" + eval.meanAbsoluteError() + " ;RMSE_global;" + eval.rootMeanSquaredError() + "  ";
      
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ;piece;" + piece + "  ";
                ArrayList<IImage> BimageList = imagecopie.sliceImage(piece);

                ArrayList<IImage> OimageList = new IImage(image.img).sliceImage(piece);

                IImage imageFinale2 = new IImage().RecollerMorceau(BimageList);                
                

                ArrayList<IImage> pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                for (int i = 0; i < BimageList.size(); i++) {

                   // System.out.println(BimageList.get(i).getPixelMatrix().length + "---" + BimageList.get(i).getPixelMatrix()[0].length);

                    Map map2 = WekaUtils.Evaluate( (int[][]) OimageList.get(i).getPixelMatrix(), (int[][]) BimageList.get(i).getPixelMatrix(),"RT");

                    pimageList.add(new IImage((int[][]) map2.get("img")));

                    pevalList.add((Evaluation) map2.get("eval"));
                }

                IImage imageFinale = new IImage().RecollerMorceau(pimageList);
                ArrayList<Double> ListValue= new ArrayList();
                double[] stat = Utils.getStat(pevalList);

                debug += "; MAE_decoupe;" + piece + ";" + stat[0] + ";" + piece + ";" + stat[1] + "  ";
                //debug += "\n";                         //System.out.println(debug);
                imageFinale.SaveImage(Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_CART_" + piece + ".jpg");

            }
            //mr.put("statList", statList);

            debug += "\n";
        }
     
        FileWriter writer = new FileWriter(fileName + ".txt",true);
        writer.write(debug);
        writer.close();
    }
   
/////////////////////////////////////////////////////////////////////////////////////////////////
    ////------------------------Test  ------------------------------------------------------ 
    public void TestValue(String fileName, String path) throws IOException, Exception {
        //System.out.println(Utils.default_dataPath + "" + fileName + ".jpg");
        int[][] matrixImage = Utils.LoadImage(fileName, path);
        //System.out.println(Utils.default_dataPath + "/" + fileName + ".jpg");
        String debug = " [" + fileName + "  ";
        for (double percent : percents) {

            debug += " | percent=" + percent + "  ";
            int[][] matrixImageCopy = Utils.cloneArray(matrixImage);

            int[][] matrixImageBruitee = Utils.GenerateBruit(matrixImageCopy, percent);
            int percent2 = (int) (percent * 100);
            Utils.SaveImageWithNA(matrixImageBruitee, Utils.default_imagePath + "/" + fileName + "_" + percent2 + ".jpg");

            /*Map map = ExecuteValue(fileName, matrixImage, matrixImageBruitee, percent2, 0, 0);
            Evaluation eval = (Evaluation) map.get("eval");

            debug += " MAE_global= " + eval.meanAbsoluteError() + "  RMSE_global=" + eval.rootMeanSquaredError() + "  ";
            */
            // System.out.println(eval.toSummaryString());
            ArrayList statList = new ArrayList();
            for (int piece : pieces) {

                debug += " ; {piece=" + piece + "  ";
                ArrayList OimageList = Utils.sliceImage(matrixImage, piece);
                ArrayList BimageList = Utils.sliceImage(matrixImageBruitee, piece);
                ArrayList pimageList = new ArrayList();
                ArrayList<Evaluation> pevalList = new ArrayList();
                System.out.println("Size=" + BimageList.size());

                for (int i = 0; i < BimageList.size(); i++) {

                    Map map2 = ExecuteValue(fileName, (int[][]) OimageList.get(i), (int[][]) BimageList.get(i), percent2, piece, i + 1);
                    pimageList.add((int[][]) map2.get("img"));
                    pevalList.add((Evaluation) map2.get("eval"));
                }
                int[][] imageComplete = Utils.RecollerMorceau(pimageList, piece);

                double[] stat = Utils.getStat(pevalList);

                debug += " MAE_decoupe" + piece + "= " + stat[0] + "  RMSE_decoupe" + piece + "=" + stat[1] + "  ";
                debug += " }";
                //System.out.println(debug);

                //statList.add(stat);
                Utils.SaveImageWithNA(imageComplete, Utils.default_imagePath + "/" + fileName + "_" + percent2 + "_rf_" + piece + ".jpg");
            }
            //mr.put("statList", statList);

            debug += "|";
        }
        // System.out.println(debug);
        // File file = new File(fileName+".txt");
        FileWriter writer = new FileWriter(fileName + ".txt");
        writer.write(debug);
        writer.close();
    }

///////////////////////////////////////////////////////////////////////////////
/////////Oesooo
    public Map ExecuteValue(String fileName, int[][] img, int[][] imgBruitee, int percent, int nbpiece, int idpiece) throws IOException, Exception {
        double[][] regressionData = Utils.createVariables(imgBruitee, radius1, radius2, radius3, minElementR1, minElementR2, minElementR2);
        //int ;
/*
       double[][] trainingData = Utils.getTrainingData(regressionData);

        String sTraining = nbpiece == 0 ? "training" : "training_" + nbpiece + "_" + idpiece;
        String sTest = nbpiece == 0 ? "test" : "test_" + nbpiece + "_" + idpiece;

        String csvFileName_training = Utils.default_dataPath + "/" + fileName + "_" + percent + "_" + sTraining + ".csv";
        String csvFileName_test = Utils.default_dataPath + "/" + fileName + "_" + percent + "_" + sTest + ".csv";
        CSVUtils.writecsv(trainingData, csvFileName_training);

      //  Utils.writecsv(variables, csvFileName_training);
        
        double[][] testData = Utils.getTestData(regressionData);
        int i = 0;
        for (double[] instance : testData) {
            int x = (int) instance[0];
            int y = (int) instance[1];
            testData[i++][2] = img[x - 1][y - 1];

        }

        CSVUtils.writecsv(testData, csvFileName_test);
        // System.exit(0);
 
        Evaluation eval = wekaFunction(fileName, percent, nbpiece, idpiece);
        
        */
       // return WekaUtils.EvaluateRF()
     //Evaluation eval= WekaUtils.EvaluateRF(regressionData,  img) ;
       
        //Evaluation eval = this.wekaFunction(fileName + "_" + percent, "data/");
        ///////////////////////////////
        Map map = new HashMap<>();

       /* if (nbpiece == 0) {
            this.SavePrediction(img, testData, eval, fileName, "rf", percent);
        } else {
            this.SavePrediction(img, testData, eval, fileName, "rf", percent, nbpiece, idpiece);
        }*/
       // map.put("eval", eval);
        map.put("img", img);
        return map;
    }

    public ArrayList<double[]> getBruitFromCsv(String FileName) throws FileNotFoundException {

        ArrayList<String[]> elementsF = CSVUtils.CSVReader(FileName);
        ArrayList<double[]> positionList = new ArrayList();
        int j = 0;
        for (String[] line : elementsF) {

            if (j != 0) {
                double[] position = new double[16];
                position[0] = Double.parseDouble(line[0]);
                position[1] = Double.parseDouble(line[1]);
                position[2] = line[2].equals("NA") ? -1 : Double.parseDouble(line[2]);
                for (int i = 3; i < 16; i++) {
                    position[i] = Utils.arrondi(Double.parseDouble(line[i]));
                }
                positionList.add(position);

            }
            j++;
        }
        return positionList;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
    public int[][] SavePrediction(int[][] matrixImageCopy, double[][] testData, Evaluation eval, String fileName, String rfunction, int percent, int nbpiece, int numpiece) {

        rfunction = rfunction + "_" + nbpiece + "_" + numpiece;
        return this.SavePrediction(matrixImageCopy, testData, eval, fileName, rfunction, percent);
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////

    public int[][] SavePrediction(int[][] matrixImageCopy, double[][] testData, Evaluation eval, String fileName, String rfunction, int percent) {

        ArrayList<Prediction> Result = eval.predictions();
        int i = 0;
        for (Prediction predicted : Result) {
            int x = (int) testData[i][0], y = (int) testData[i++][1];
            //System.out.println( " X="+x+" Y="+y+" "+matrixImageCopy[x-1][y-1]+"  "+predicted.predicted());

            // System.exit(0);
            matrixImageCopy[x - 1][y - 1] = (int) predicted.predicted();
        }
        Utils.SaveImageWithNA(matrixImageCopy, Utils.default_imagePath + "/" + fileName + "_" + percent + "_" + rfunction + ".jpg");
        return matrixImageCopy;
    }

    
    
    
    
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///Learning 
    public Evaluation wekaFunction(String fileName, int percent, int nbpiece, int idpiece) throws FileNotFoundException, IOException, Exception {

        String sTraining = nbpiece == 0 ? "training" : "training_" + nbpiece + "_" + idpiece;
        String sTest = nbpiece == 0 ? "test" : "test_" + nbpiece + "_" + idpiece;

        String csvFileName_training = Utils.default_dataPath + "/" + fileName + "_" + percent + "_" + sTraining + ".csv";
        String csvFileName_test = Utils.default_dataPath + "/" + fileName + "_" + percent + "_" + sTest + ".csv";

        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(",");
        loader.setSource(new File(csvFileName_training));
        Instances trainingdata = loader.getDataSet();
        // System.out.println("--------"+csvFileName_training+"-------"+csvFileName_test+"");
        CSVLoader loader2 = new CSVLoader();
        loader2.setFieldSeparator(",");
        loader2.setSource(new File(csvFileName_test));
        Instances testdata = loader2.getDataSet();
        final Classifier classifier = new RandomForest();
        trainingdata.setClassIndex(2);
        testdata.setClassIndex(2);
        classifier.buildClassifier(trainingdata);
        Evaluation eval = new Evaluation(trainingdata);
        eval.evaluateModel(classifier, testdata);
        // ArrayList<Prediction> predictions = eval.predictions();
        return eval;
    }

    public static void main(String[] args) {

        String pathName = "rimages/";
        String regFunction="RF";
        if (args.length != 0) {
            pathName = args[0];
            if(args.length>0) regFunction=args[1];
        }
        new ImageInpaint(pathName,regFunction);

    }

}
