/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import Inpainting.ImageInpaint;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Double.NaN;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;

/**
 *
 * @author koueya
 */
public class Utils {

    public static String default_imagePath = "graphics/";
    public static String default_dataPath = "data/";
    public static final int radius1 = 10;
    public static final int radius2 = 20;
   public static final int radius3 = 30 ;
    public static final int minElementR1 = 2, minElementR2 = 5, minElementR3 = 10;
/////////////////////////////////////////
//Chargement d'une image à partir d'un fichier
    public static int[][] LoadImage(String fileName) throws IOException {
        return LoadImage(fileName, default_dataPath);
    }
    /////////////////////////////////////////
//Chargement d'une image à partir d'un fichier

    public static int[][] LoadImage(String fileName, String dataPath) throws IOException {
        BufferedImage img = ImageIO.read(new File(dataPath + "/" + fileName + ".jpg"));
        return asMatrix(img);
    }
////////////////////////////////////////////////////////////////////////////////////////
////Sauvegarder les morceau d'images

    public static void outputMorceauImage(ArrayList<int[][]> liste_morceau, String fileName) {
        outputMorceauImage(liste_morceau, fileName, default_imagePath);
    }

////////////////////////////////////////////////////////////////////////////////////////
////Sauvegarder les morceau d'images
    public static void outputMorceauImage(ArrayList<int[][]> liste_morceau, String fileName, String imagePath) {

        if (imagePath != null) {
            fileName = imagePath + "/" + fileName;
        }
        int i = 0;
        for (int[][] morceau : liste_morceau) {
            //fileName=fileName+"_"+i+".jpg";
            //System.out.println("Morceau "+i+" === "+morceau[0].length);
            SaveImageWithNA(morceau, morceau.length, morceau[0].length, fileName + "_" + i + ".jpg");
            i++;
        }
    }

////////////////////////////////////////////////////////////////////////////////////
////Recoller les morceau d'image
    public static int[][] RecollerMorceau2(ArrayList<int[][]> liste_morceau, int nbpiece) {

        int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        //int j=0;
        int w = 0;
        int h = 0;

        for (int t = 0; t < nbpiece_per_row; t++) {
            w += liste_morceau.get(t)[0].length;// ncol(liste_morceau[[t]])+nco

            h += liste_morceau.get(t).length;
        }
        int NB_R = h / nbpiece_per_row;
        int NB_C = w / nbpiece_per_row;
        int y = 0, L = 0;
        int imageMatrix[][] = new int[h][w];

//imageMatrix<-matrix(, nrow =  nro, ncol =  nco)
        for (int t = 0; t < nbpiece_per_row; t++) {

            int heigth_si = NB_R;

            if (heigth_si * nbpiece_per_row != imageMatrix.length && t == 0) {
                heigth_si = heigth_si + (imageMatrix.length - heigth_si * nbpiece_per_row);

            }
            int b = 0;

            for (int i = 0; i < nbpiece_per_row; i++) {

                int width_si = NB_C; //imageMatrix[0].length/nbpiece_per_row;

                if ((width_si * nbpiece_per_row != imageMatrix[0].length) && i == 0) {
                    width_si = width_si + (imageMatrix[0].length - width_si * nbpiece_per_row);

                }

                //#Result[y]<- matrix(, nrow = NB_R, ncol = NB_C)
                //int[][] Moceau = new int[heigth_si][width_si];

                for (int j = 0; j < liste_morceau.get(y).length; j++) {
                    for (int k = 0; k < liste_morceau.get(y)[0].length; k++) {

                        // Moceau[j][k] = 
                        imageMatrix[j + L][k + b] = liste_morceau.get(y)[j][k];
                    }
                }
                b = b + width_si;
                y++;

            }
            L = L + heigth_si;
             System.out.println(" L="+L+"  b="+b);
        }   
       
        

        return imageMatrix;
    }

  /////////////////////////////////////////////////////////
/// #Fonction de découpage 
    public static int[][] RecollerMorceau(ArrayList<int[][]> liste_morceau , int nbpiece) {

        int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        int w=0,h=0;        
        
        for (int t = 0; t < nbpiece_per_row; t++) {
            h += liste_morceau.get(t)[0].length;// ncol(liste_morceau[[t]])+nco
            System.err.println(" ID="+t*nbpiece_per_row+" Lenggth="+liste_morceau.get(t*nbpiece_per_row).length+"  ");
            w += liste_morceau.get(t*nbpiece_per_row).length;         
          
        }
    //  System.exit(0);
        int imageMatrix[][] = new int[w][h];        
      
        int y = 0, L = 0;
       
        for (int t = 0; t < nbpiece_per_row; t++) {
            int heigth_si = liste_morceau.get(y).length;
            int b = 0;            
            for (int i = 0; i < nbpiece_per_row; i++) {

                int width_si =  liste_morceau.get(y)[0].length;               
                for (int j = 0; j < heigth_si; j++) {
                    for (int k = 0; k < width_si; k++) {                       
                        imageMatrix[j + L][k + b] = liste_morceau.get(y)[j][k];
                    }
                }
                b = b + width_si;
                y++;
             }
            L = L + heigth_si;
            
        }       
         return imageMatrix;
    }  
    
   
   public static int[][] getPositionMorceau(double [][]regressionData,ArrayList<int[][]> liste_morceau ,  int nbpiece, int[][]morceau,int indice) {

        int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        int w=0,h=0;        
        
        for (int t = 0; t < nbpiece_per_row; t++) {
            h += liste_morceau.get(t)[0].length;// ncol(liste_morceau[[t]])+nco
            System.err.println(" ID="+t*nbpiece_per_row+" Lenggth="+liste_morceau.get(t*nbpiece_per_row).length+"  ");
            w += liste_morceau.get(t*nbpiece_per_row).length;         
          
        }
    //  System.exit(0);
        int imageMatrix[][] = new int[w][h];        
      
        int y = 0, L = 0;
       
        for (int t = 0; t < nbpiece_per_row; t++) {
            int heigth_si = liste_morceau.get(y).length;
            int b = 0;            
            for (int i = 0; i < nbpiece_per_row; i++) {

                int width_si =  liste_morceau.get(y)[0].length;               
                for (int j = 0; j < heigth_si; j++) {
                    for (int k = 0; k < width_si; k++) {                       
                        imageMatrix[j + L][k + b] = liste_morceau.get(y)[j][k];
                    }
                }
                b = b + width_si;
                y++;
             }
            L = L + heigth_si;
            
        }       
         return imageMatrix;
    }  
       

////////////////////////////////////////////////////////////////////////////////
//Generation de bruit dans une image
    public static int[][] GenerateBruit(int img[][], ArrayList<int[]> vecteur) {      
     
   
        for(int i[]: vecteur){
        
       // rand.forEach(i -> {
           // double rowD = i %nbcol;
            int row = i[0];
            int col = i[1]; 
            
            img[row][col] = -1;        
        
        }
     return img;
  }
    
/////////////////////////////////////////////////////////
/// #Fonction de découpage 
    public static ArrayList<int[][]> sliceImage(int imageMatrix[][], int nbpiece) {

        int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        int NB_R = (int) imageMatrix.length / nbpiece_per_row;
        int NB_C = imageMatrix[0].length / nbpiece_per_row;
        int y = 0, L = 0;
        ArrayList<int[][]> Result = new ArrayList();
        for (int t = 0; t < nbpiece_per_row; t++) {

            int heigth_si = NB_R;

            if (heigth_si * nbpiece_per_row != imageMatrix.length && t == 0) {
                heigth_si = heigth_si + (imageMatrix.length - heigth_si * nbpiece_per_row);

            }
            int b = 0;

            for (int i = 0; i < nbpiece_per_row; i++) {

                int width_si = NB_C; //imageMatrix[0].length/nbpiece_per_row;

                if ((width_si * nbpiece_per_row != imageMatrix[0].length) && i == 0) {
                    width_si = width_si + (imageMatrix[0].length - width_si * nbpiece_per_row);

                }

                //#Result[y]<- matrix(, nrow = NB_R, ncol = NB_C)
                int[][] Moceau = new int[heigth_si][width_si];
                //System.out.println(" NC="+NB_C);
                for (int j = 0; j < heigth_si; j++) {
                    for (int k = 0; k < width_si; k++) {

                        Moceau[j][k] = imageMatrix[j + L][k + b];
                    }
                }
                b = b + width_si;        
                Result.add(Moceau);
                //#Result[y]$data<-Moceau

                y++;

            }
            L = L + heigth_si;
        }

        return Result;

    }
    
    
    /////////////////////////////////////////////////////////
/// #Fonction de découpage 
    public static ArrayList<int[][]> sliceImage(int imageMatrix[][], int nbpiece, Map position) {

        int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        int NB_R = (int) imageMatrix.length / nbpiece_per_row;
        int NB_C = imageMatrix[0].length / nbpiece_per_row;
        int y = 0, L = 0;
        ArrayList<int[][]> Result = new ArrayList();
        for (int t = 0; t < nbpiece_per_row; t++) {

            int heigth_si = NB_R;

            if (heigth_si * nbpiece_per_row != imageMatrix.length && t == 0) {
                heigth_si = heigth_si + (imageMatrix.length - heigth_si * nbpiece_per_row);

            }
            int b = 0;

            for (int i = 0; i < nbpiece_per_row; i++) {

                int width_si = NB_C; //imageMatrix[0].length/nbpiece_per_row;

                if ((width_si * nbpiece_per_row != imageMatrix[0].length) && i == 0) {
                    width_si = width_si + (imageMatrix[0].length - width_si * nbpiece_per_row);

                }

                //#Result[y]<- matrix(, nrow = NB_R, ncol = NB_C)
                int[][] Moceau = new int[heigth_si][width_si];
                //System.out.println(" NC="+NB_C);
                for (int j = 0; j < heigth_si; j++) {
                    for (int k = 0; k < width_si; k++) {

                        Moceau[j][k] = imageMatrix[j + L][k + b];
                    }
                }
                   
                Result.add(Moceau);
                int pos[]= {L,b};
                position.put(y, pos);
                //pos[]
                b = b + width_si;     
                //#Result[y]$data<-Moceau

                y++;

            }
            L = L + heigth_si;
        }

        return Result;

    }
    
    ///////////////////////////////////////////////////////////////////////////////
///Sauvegarde l'image contenant les valeurs nulles

    public static void SaveImageWithNA(int[][] img, String fileName) {
        SaveImageWithNA( img, img.length, img[0].length, fileName);
        
    }
///////////////////////////////////////////////////////////////////////////////
///Sauvegarde l'image contenant les valeurs nulles

    public static void SaveImageWithNA(int[][] img, int nbrow, int nbcol, String fileName) {
        // BufferedImage image2 = new BufferedImage(nbcol, nbrow, BufferedImage.TYPE_BYTE_GRAY);
        BufferedImage image2 = new BufferedImage(nbcol, nbrow, BufferedImage.TYPE_INT_RGB);

        IntStream.range(0, nbrow).parallel().forEach(
                i -> {
                    //int k=0;
                    IntStream.range(0, nbcol).forEach(
                            j -> {
                                Color Mycolor;
                                if (img[i][j] == -1) {

                                    //  k=k+1; 
                                    //System.out.println("+++"+matrixImage2[i][j]);
                                   Mycolor= new Color(255, 255, 255);
                                    //img[i][j] = 255;
                                }else Mycolor= new Color(img[i][j], img[i][j], img[i][j]);
                                 
                                image2.setRGB(j, i, Mycolor.getRGB());
                            }
                    );

                }
        );

        try {
            // retrieve image
            // BufferedImage bi = getMyImage();
            File outputfile = new File(fileName);
            ImageIO.write(image2, "jpg", outputfile);
        } catch (IOException e) {

        }

    }

    ////////////////////////////////////////////////////////////////////////////////
//Generation de bruit dans une image
    public static int[][] GenerateBruit(int img[][],  double percent) {
     
        GenerateBruit(img, img.length, img[0].length,percent);
        //  IntStream.range(1, 5).parallel().forEach(i -> heavyOperation());

        return img;
    }
  
    public static int[] generateRange(int pCount, int pMin, int pMax) {
    int min = pMin < pMax ? pMin : pMax;
    int max = pMax > pMin ? pMax : pMin;
    int resultArr []= new int[pCount];
        ArrayList<Integer> list = new ArrayList<Integer>();
        for (int i=min; i<max; i++) {
            list.add(new Integer(i));
        }
        Collections.shuffle(list);
        for (int i=0; i<pCount; i++) {
            resultArr[i]=list.get(i);          
        }
    
    return resultArr;
  }
////////////////////////////////////////////////////////////////////////////////
//Generation de bruit dans une image
    public static int[][] GenerateBruit(int img[][], int nbrow, int nbcol, double percent) {
       
        double nbMissing = percent * nbrow * nbcol;
        Double nbm = percent * nbrow * nbcol;
        
     
        IntStream rand = new Random().ints(nbm.intValue(), 0, nbrow * nbcol);
       // 
        //IntStream rand = new Random().ints(12, 0, 30);
        int number[]=generateRange(nbm.intValue(),0,nbrow * nbcol);
        int j=0;
        for(int i: number){
        
       // rand.forEach(i -> {
           // double rowD = i %nbcol;
            int row = (int) (i /nbcol);
            int col = i - row * nbcol;
          
            if(img[row][col]==-1){
                
            
                 j++;
            }
            img[row][col] = -1; 
           
        
        }
     

        return img;
    }

    ////////////////////////////////////////////////////////////////////////////////
//Generation de bruit dans une image
    public static int[][] GenerateBruit(int img[][], int []vecteur) {      
     
     int nbcol= img[0].length;
 
   
        int j=0;
        for(int i: vecteur){
        
       // rand.forEach(i -> {
           // double rowD = i %nbcol;
            int row = (int) (i /nbcol);
            int col = i - row * nbcol;
          
            if(img[row][col]==-1){
                
                //System.out.println(" COl =["+row+" ],["+col+"]");
                 j++;
            }
            img[row][col] = -1; 
            //System.out.println(" NB_J"+i);
        
        }
     return img;
  }
    
  public static double[] getStat( ArrayList<Evaluation> evalList){
   double MAE=0,RAE=0,RMSE=0,RRAE=0,RMPSE=0, result[]=new double[2];
   int i=0;
  //  int y=0;
   for(Evaluation eval : evalList){
   
       ArrayList<Prediction> predictions= eval.predictions();
      if(predictions!=null)
       for(Prediction pred: predictions){
           
           RMSE+=(pred.predicted()-pred.actual())*(pred.predicted()-pred.actual());
           MAE+=Math.abs(pred.predicted()-pred.actual());
           
         //  if(y==3)System.out.println(pred.actual()+";"+pred.predicted());
          // eval.r
         i++;
       }
     //  y++;
       //eval.toSummaryString(default_imagePath, true)
     //System.out.println(eval.toSummaryString());  
   }    
     //  System.out.println("----------------------------------------------------------------------");
   
  i=i==0?1:i;
   // System.out.println(" I="+i);
       MAE=MAE/i;
       RMSE=Math.sqrt(RMSE/i);
       result[0]=MAE;
       result[1]=RMSE;
       return result;
  }
public static int[][] cloneArray(int[][] src) {
    int length = src.length;
    int[][] target = new int[length][src[0].length];
    for (int i = 0; i < length; i++) {
        System.arraycopy(src[i], 0, target[i], 0, src[i].length);
    }
    return target;
}
public static double[][] cloneArrayDouble(double[][] src) {
    int length = src.length;
    double[][] target = new double[length][src[0].length];
    for (int i = 0; i < length; i++) {
        System.arraycopy(src[i], 0, target[i], 0, src[i].length);
    }
    return target;
}


////////////////////////////////////////////////////////////////////////////////////////////
//Recupérer l'images comme une matrice
    public static int[][] asMatrix(BufferedImage img) {

        int w = img.getWidth();
        int h = img.getHeight();
        int image[][] = new int[h][w];
// Safe cast as img is of type TYPE_BYTE_GRAY 
        WritableRaster wr = img.getRaster();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                image[y][x] = wr.getSample(x, y, 0);
            }
        }

        return image;

    }
    ////////////////////////////////////////////////////////////////////////////////
//Retourne le training Set

    public static double[][] getTrainingData(double dataFrame[][]) {
        int nbrow = dataFrame.length;
        ArrayList<double[]> tableau = new ArrayList();
        IntStream.range(0, nbrow).forEach(
                i -> {

                    if (dataFrame[i][2] != -1) {
                        tableau.add(dataFrame[i]);
                    }

                }
        );
        double[][] tab = new double[tableau.size()][16];
        int i = 0;
        for (double[] el : tableau) {
            tab[i++] = el;
        }
        return tab;
    }
////////////////////////////////////////////////////////////////////////////////
//Retourne le training Set

    public static double[][] getTestData(double dataFrame[][]) {
        //int image2[][] = new int[nbrow][nbcol];
        int nbrow = dataFrame.length;
        ArrayList<double[]> tableau = new ArrayList();
//double[][] tableau = new double[nbrow][16];
        IntStream.range(0, nbrow).forEach(
                i -> {

                    if (dataFrame[i][2] == -1) {
                        tableau.add(dataFrame[i]);
                    }

                }
        );
        // tableau.add(1);

        double[][] tab = new double[tableau.size()][16];
        int i = 0;
        for (double[] el : tableau) {
            tab[i++] = el;
        }
        return tab;
        // return tablea
    }

////////////////////////////////////////////////////////////////////////////////
//Ecriture dans le fichier CSV

 //////////////////////////////////////////
    
    
    
        public static double[][] createVariables(int[][] image1, int radius1, int radius2, int radius3,int minElementR1,int minElementR2,int minElementR3) {
          
            return createVariables(image1, image1.length, image1[0].length, radius1, radius2, radius3, minElementR1,minElementR2,minElementR3);
            
        }
/**
 * 
 * @param image1
 * @param nbrows
 * @param nbcols
 * @param radius1
 * @param radius2
 * @param radius3
 * @return 
 */
            /**
     * Creation des variables predictives
     *
     * @param radius1 rayon du premier contour
     * @param radius2 rayon du second contour
     * @param radius3 rayon du troisieme contour
     * @param minElementR1 nombre min de pixel dans le quadran1
     * @param minElementR2 nombre min de pixel dans le quadran2
     * @param minElementR3 nombre min de pixel dans le quadran3
     */

    public static double[][] createVariables(int[][] image1, int nbrows, int nbcols, int radius1, int radius2, int radius3,int minElementR1,int minElementR2,int minElementR3) {

        //int nbrows = image1.getWidth(), nbcols = image1.getHeight();
        
      //  System.out.println(" NBrow="+nbrows+" NBcol="+ nbcols);
        double image2[][] = new double[nbrows * nbcols][16];       
        int z = 0;

        IntStream.range(0, nbrows).forEach(
                i -> {

                    IntStream.range(0, nbcols).forEach(
                            j -> {
                                //for (int j = 0; j < nbcols; j++) {

                                double mean1 = 0, mean2 = 0, mean3 = 0, sd1 = 0, sd2 = 0, sd3 = 0, wmean1 = 0, wmean2 = 0, wmean3 = 0, trend_m1, trend_m2, trend_w1, trend_w2;
                                int  minE1=0,minE2=0,minE3=0;
                                
//On va devoir stocker les valeurs pixel et distance à utiliser pour calculer les valeurs des variables
                                ArrayList pixelList1 = new ArrayList(), distanceList1 = new ArrayList();
                                ArrayList pixelList2 = new ArrayList(), distanceList2 = new ArrayList();
                                ArrayList pixelList3 = new ArrayList(), distanceList3 = new ArrayList();
                                
                                  //System.out.println("Radius="+radius1);
                                for (int k = 0; k <= radius3; k++) {

                                    for (int t = 0; t <=radius3; t++) {

                                        //Distance
                                        double dist = Math.sqrt(t * t + k * k);

                                      
                                        
                                         
                                        if (dist > 0 && (dist <= radius3 /*|| minE3<minElementR3*/)) {
                                           
                                            if (((i + k) < nbrows) && ((j + t) < nbcols) && (image1[i + k][j + t] != -1)) {
                                                 minE3++;
                                                pixelList3.add(image1[i + k][j + t]);
                                                distanceList3.add(dist);

                                                if (dist <= radius1/*|| minE1<minElementR1*/) {
                                                    minE1++;
                                                    pixelList1.add(image1[i + k][j + t]);
                                                    distanceList1.add(dist);                                                   
                                                }
                                                if (dist <= radius2 /*|| minE2<minElementR2*/) {
                                                    minE2++;
                                                    pixelList2.add(image1[i + k][j + t]);
                                                    distanceList2.add(dist); 
                                                // if(( i + 1)==3 && (j + 1)==160)     System.out.println(" (i+"+k+" , j+"+t+")"+image1[i + k][j + t]);

                                                }

                                            }
                                            
                                            if (((i - k) >= 0) && ((j + t) < nbcols) && (k>0) && (image1[i - k][j + t] != -1)) {
                                                

                                                 minE3++;
                                                pixelList3.add(image1[i - k][j + t]);
                                                distanceList3.add(dist);

                                                if (dist <= radius1/*|| minE1<minElementR1*/) {
                                                    minE1++;
                                                    pixelList1.add(image1[i - k][j + t]);
                                                    distanceList1.add(dist);

                                                }
                                                if (dist <= radius2 /*|| minE2<minElementR2*/) {
                                                  minE2++;
                                                    pixelList2.add(image1[i - k][j + t]);
                                                    distanceList2.add(dist);
                                                    //      if(( i + 1)==3 && (j + 1)==160) System.out.println(" (i-"+k+" , j+"+t+")"+image1[i - k][j + t]);

                                                }
                                            }
                                            if (((i - k) >= 0) && ((j - t) >= 0) && (k>0) && (t>0) && (image1[i - k][j - t] != -1)) {
                                                 minE3++;
                                                pixelList3.add(image1[i - k][j - t]);
                                                distanceList3.add(dist);

                                                if (dist <= radius1 /*|| minE1<minElementR1*/) {
                                                    minE1++;
                                                    pixelList1.add(image1[i - k][j - t]);
                                                    distanceList1.add(dist);

                                                }
                                                if (dist <= radius2 /*|| minE2<minElementR2*/) {
                                                    minE2++;
                                                    pixelList2.add(image1[i - k][j - t]);
                                                    distanceList2.add(dist);
                                                  //   if(( i + 1)==3 && (j + 1)==160)  System.out.println(" (i-"+k+" , j-"+t+")"+image1[i - k][j - t]);


                                                }

                                            }

                                            if (((i + k) < nbrows) && ((j - t) >= 0) && (t>0) && (image1[i + k][j - t] != -1)) {
                                                 minE3++;
                                                pixelList3.add(image1[i + k][j - t]);
                                                distanceList3.add(dist);
                                                 

                                                if (dist <= radius1 /*|| minE1<minElementR1*/) {
                                                    minE1++;
                                                    pixelList1.add(image1[i + k][j - t]);
                                                    distanceList1.add(dist);
                                                }
                                                if (dist <= radius2 /*|| minE2<minElementR2*/) {
                                                    minE2++;
                                                    pixelList2.add(image1[i + k][j - t]);
                                                    distanceList2.add(dist);
                                                   // if(( i + 1)==3 && (j + 1)==160) System.out.println(" (i+"+k+" , j-"+t+")"+image1[i + k][j - t]);
                                                }

                                            }

                                        }
                                    }

                                }
                                //if(minE3<10) System.out.println(" Error Code "+minE3);
                                ///////////////////////////////////
                                //Mean Variable
                                mean1 = mean(pixelList1);
                                mean2 = mean(pixelList2);
                                mean3 = mean(pixelList3);
                                /////////////////////////
                                wmean1 = wmean(pixelList1, distanceList1);
                                wmean2 = wmean(pixelList2, distanceList2);
                                wmean3 = wmean(pixelList3, distanceList3);
                                ////////////////////////////////////
                                double m2 = mean2 == 0 ? 1 : mean2, m3 = mean3 == 0 ? 1 : mean3;
                                double w2 = wmean3 == 0 ? 1 : wmean2, w3 = wmean3 == 0 ? 1 : wmean3;
                                trend_m1 = mean1 / m2;
                                trend_m2 = mean2 / m3;
                                trend_w1 = wmean1 / w2;
                                trend_w2 = wmean2 / w3;
                                ///////////////////////////////

                                //Standar deviation Variable
                                sd1 = sd(pixelList1);
                                sd2 = sd(pixelList2);
                                sd3 = sd(pixelList3);
                                //#cat("i=",i," j=",j)
                                // image2[i][j] = mean3;
                                //  System.out.println("Aaron");
                                int indice = i * nbcols + j;
                                image2[indice][0] = i + 1;
                                image2[indice][1] = j + 1;
                                image2[indice][2] = image1[i][j];
//LibSVM classifier = new LibSVM();
                                image2[indice][3] = arrondi(mean1);
                                image2[indice][4] = arrondi(mean2);
                                image2[indice][5] = arrondi(mean3);

                                image2[indice][6] = arrondi(trend_m1);
                                image2[indice][7] = arrondi(trend_m2);

                                image2[indice][8] = arrondi(wmean1);
                                image2[indice][9] = arrondi(wmean2);
                                image2[indice][10] = arrondi(wmean3);

                                image2[indice][11] = arrondi(trend_w1);
                                image2[indice][12] = arrondi(trend_w2);

                                image2[indice][13] = arrondi(sd1);
                                image2[indice][14] = arrondi(sd2);
                                image2[indice][15] = arrondi(sd3);
                                
                                
                               /* if(( i + 1)==3 && (j + 1)==160)
                                {
                                for(int k=0;k<pixelList2.size();k++){
                                    System.out.println(" "+pixelList2.get(k));
                                } 
                                System.out.println(pixelList2.size()+" ----"+minE2+"----pixel="+ image1[i][j]+" Mean="+arrondi(mean2));
                                } */
                               // System.exi(0);

                            });
                });

        return image2;
    }
    
    public static ArrayList listerRepertoire(File repertoire) {

        String[] listefichiers;
        ArrayList liste= new ArrayList();
        int i;
        listefichiers = repertoire.list();
        for (i = 0; i < listefichiers.length; i++) {
            if (listefichiers[i].endsWith(".jpg") == true) {

                liste.add(listefichiers[i].substring(0, listefichiers[i].length() - 4));
            }
        }
        
        return liste;
    }
/////////////////////////////////////
//Arrondi
    public static double arrondi(double a) {
        
        return a;
       /* if(a==NaN) return NaN;
        BigDecimal bd = new BigDecimal(a);
        bd = bd.setScale(5, BigDecimal.ROUND_DOWN);
        a = bd.doubleValue();
        return a; */
    }
     public static double arrondi(double a, int nbchiffre) {
        if(a==NaN) return NaN;
        BigDecimal bd = new BigDecimal(a);
        bd = bd.setScale(nbchiffre, BigDecimal.ROUND_DOWN);
        a = bd.doubleValue();
        return a;
    }

    public static double mean(ArrayList list) {
        int i = 0;
        double mean = 0;
        for (i = 0; i < list.size(); i++) {
            mean += (int) list.get(i);
        }

        return mean / i;
    }

    public static double sd(ArrayList<Integer> list) {
        //int i = 0;
        double mean = mean(list), x = 0, sd;
        for (int i : list ) {
            x += Math.pow(i-mean, 2);
        }

        return Math.sqrt(x / (list.size()-1));
    }

    public static double wmean(ArrayList list, ArrayList dist) {
        int i = 0;
        double mean = 0, sdist = 0;
        for (i = 0; i < list.size(); i++) {
            mean += ((int) list.get(i) * (1 / (double) dist.get(i)));
            sdist += 1 / (double) dist.get(i);
        }

        return mean / sdist;
    }
    
   public  static BufferedImage deepImageCopy(BufferedImage bi) {
        ColorModel cm = bi.getColorModel();
        boolean isAlphaPremultiplied = cm.isAlphaPremultiplied();
        WritableRaster raster = bi.copyData(null);
        return new BufferedImage(cm, raster, isAlphaPremultiplied, null);
    }

}


