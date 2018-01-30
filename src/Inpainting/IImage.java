package Inpainting;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;
import utils.Utils;
import static utils.Utils.arrondi;
import static utils.Utils.asMatrix;
import static utils.Utils.generateRange;
import static utils.Utils.mean;
import static utils.Utils.sd;
import static utils.Utils.wmean;
//import static utils.Utils.default_dataPath;

/**
 * classe représentant une image
 * @author koueya
 */
public class IImage {

    /**
     * Nombre de ligne dans l'image
     */
    private int nbrow;
    /**
     * Nombre de colonnes dans l'image
     */
    private int nbcol;
    /**
     * Buffer de l'image
     */
    BufferedImage img;
    /**
     * Matrice de l'image, indique le niveau de gris d'un pixel
     */
    private int[][] pixelMatrix;
    /**
     * Nom du fichier
     */
    private String fileName;

    /**
     * Copie originale de l'image
     */
    private IImage OriginalImageCopie;

    /**
     * Chargement d'une image à partir d'un fichier
     * @param fileName nom du fichier source
     * @throws IOException 
     */
    public void LoadImage(String fileName) throws IOException {
        LoadImage(fileName, Utils.default_dataPath);
    }

    /**
     * Constructeur par défaut d'une image
     */
    public IImage() {

    }

    /**
     * Constructeur depuis un fichier
     * @param fileName Nom du fichier source
     * @param dataPath Chemin du fichier
     * @throws IOException
     */
    public IImage(String fileName, String dataPath) throws IOException {
        this.LoadImage(fileName, dataPath);
    }

    /**
     * Constructeur depuis un fichier source
     * @param fileName nom du fichier source
     * @throws IOException 
     */
    public IImage(String fileName) throws IOException {
        this.LoadImage(fileName);
    }
    
    /**
     * Constructeur à partir d'une matrice de pixel
     * @param matrixPixel matrice de pixel (niveau de gris)
     */
    public IImage(int[][] matrixPixel) {

        if (matrixPixel.length > 0) {
           
             this.pixelMatrix = new int[matrixPixel.length][matrixPixel[0].length];
           
            this.nbrow = matrixPixel.length;
            this.nbcol = matrixPixel[0].length;
            this.img = new BufferedImage(this.nbcol, nbrow, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < this.nbcol; y++) {

                for (int x = 0; x < nbrow; x++) {
                 
                    this.pixelMatrix[x][y]= matrixPixel[x][y];
                   
                    if(matrixPixel[x][y]!=-1&&matrixPixel[x][y]>=0 && matrixPixel[x][y]<255)

                     //int val= (int)
                     this.img.setRGB(y, x, new Color(matrixPixel[x][y], matrixPixel[x][y], matrixPixel[x][y]).getRGB());
                    else this.img.setRGB(y, x, new Color(255, 255, 255).getRGB());
                }
            }
        }

    }

    /**
     * Constructeur à partir d'un buffer d'image (BufferedImage
     * @param img Buffer de l'image
     */
    public IImage(BufferedImage img) {
        this.img = Utils.deepImageCopy(img);
        this.pixelMatrix = this.asMatrix();
        this.nbrow = img.getHeight();
        this.nbcol = img.getWidth();

    }

    /**
     * Accesseur de nbrow
     * @return nombre de lignes de l'images
     */
    public int getNbrow() {
        return nbrow;
    }

    /**
     * Mutateur de nbrow
     * @param nbrow nombre de ligne de l'image
     */
    public void setNbrow(int nbrow) {
        this.nbrow = nbrow;
    }

    /**
     * Accesseur de nbcol
     * @return Nombre de colonnes de l'image
     */
    public int getNbcol() {
        return nbcol;
    }

    /**
     * Mutateur de nbcol
     * @param nbcol Nombre de colonnes de l'image
     */
    public void setNbcol(int nbcol) {
        this.nbcol = nbcol;
    }

    /**
     * Accesseur de img
     * @return Le buffer de l'image (BufferdImage)
     */
    public BufferedImage getImg() {
        return img;
    }

    /**
     * Mutateur de img
     * @param img Le buffer de l'image (BufferedImage)
     */
    public void setImg(BufferedImage img) {
        this.img = img;
    }

    /**
     * Accesseur de pixelMatrix
     * @return La matrice des pixels en niveau de gris
     */
    public int[][] getPixelMatrix() {
        return pixelMatrix;
    }

    /**
     * Mutateur de pixelMatrix
     * @param pixelMatrix La matrice des pixel en niveau de gris
     */
    public void setPixelMatrix(int[][] pixelMatrix) {
        this.pixelMatrix = pixelMatrix;
    }

    /**
     * Accesseur de OriginalImageCopie
     * @return La copie originale de l'image
     */
    public IImage getOriginalImageCopie() {
        return OriginalImageCopie;
    }

    /**
     * Mutateur de OriginalImageCopie
     * @param OriginalImageCopie La copie originale de l'image
     */
    public void setOriginalImageCopie(IImage OriginalImageCopie) {
        this.OriginalImageCopie = OriginalImageCopie;
    }

    /**
     * Chargement d'une image à partir d'un fichier
     * @param fileName nom du fichier à charger
     * @param dataPath Chemin du fichier
     * @throws IOException 
     */
    public void LoadImage(String fileName, String dataPath) throws IOException {
        this.img = ImageIO.read(new File(dataPath + "/" + fileName + ".jpg"));
        // this.img = 
        this.pixelMatrix = this.asMatrix();
        this.nbrow = img.getHeight();
        this.nbcol = img.getWidth();
        //  return asMatrix(img);
    }
    /**
     * Traduit le buffer de l'image en matrice
     * @return Matrice de pixel en niveau de gris
     */
    private int[][] asMatrix() {

        int w = img.getWidth();
        int h = img.getHeight();
        int matriximage[][] = new int[h][w];
        // Safe cast as img is of type TYPE_BYTE_GRAY 
        WritableRaster wr = img.getRaster();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                matriximage[y][x] = wr.getSample(x, y, 0);
            }
        }

        return matriximage;

    }
    /**
     * Generation des bruits dans l'image selon une loi continue
     * @param percent pourcentage de pixel manquant
     */
    public void GenerateBruit(double percent) {

        //On garde 
        OriginalImageCopie = new IImage(img);
        double nbMissing = percent * this.nbrow * this.nbcol;
        Double nbm = percent * this.nbrow * this.nbcol;
       // System.out.println("  NB_row=" + this.pixelMatrix.length+" -- "+this.nbrow+"  NB col="+this.pixelMatrix[0].length+"---"+this.nbcol);
        int number[] = generateRange(nbm.intValue(), 0, this.nbrow * this.nbcol);
        int j = 0;
        for (int i : number) {

            // rand.forEach(i -> {
            // double rowD = i %this.nbcol;
            int row = (int) (i / this.nbcol);
            int col = i - row * this.nbcol;

            this.pixelMatrix[row][col] = -1;
            this.img.setRGB(col, row, new Color(255, 255, 255).getRGB());
        }

    }
    
    public ArrayList GenerateHisto(){
        ArrayList<ArrayList> histo = new ArrayList<>();
        for(int i = 0; i < 256; i++){
            ArrayList e = new ArrayList();
            histo.add(e);
        }
        for(int i = 0; i <this.nbrow; i++){
            for(int j = 0; j < this.nbcol; j++){
                int p[] = {i,j};
                histo.get(this.pixelMatrix[i][j]).add(p);
            }
        }
        return histo;
    }
    
    public void GenerateBruitHistProp(double percent, int nbSepar){
        OriginalImageCopie = new IImage(img);
        Double nbMissingBySepar = percent *this.nbrow * this.nbcol / nbSepar;
        int range = 255 / nbSepar;
        
        //ArrayList listSepar = new ArrayList();
        
        ArrayList<ArrayList<int[]>> histo = GenerateHisto();
        
        for(int i = 0; i < nbSepar ; i++){//Parcours des groupes
            int count = 0;
            for(int j = range*i ; j < range*i + range; j++){
                count += histo.get(j).size();
            }
            nbMissingBySepar = percent * count;
            int number[] = generateRange(nbMissingBySepar.intValue(), 0, count-1);
            for (int k : number){
                int l = 0; int cpt = 0;
                while(cpt+histo.get(l).size()-1 < k){
                    cpt += histo.get(l).size()-1;
                    l++;
                }
                int p[];
                p = histo.get(l).get(k-cpt);
                int row = p[0];
                int col = p[1];
                this.pixelMatrix[row][col] = -1;
                this.img.setRGB(col, row, new Color(255, 255, 255).getRGB());
            }
        }
        
    }
    
    /**
     * Generation des bruits dans l'image selon un loi Gaussienne
     * @param percent pourcentage de pixel manquant
     */
    public void GenerateBruitGaussian(double percent) {

        //On garde 
        OriginalImageCopie = new IImage(img);
        double nbMissing = percent * this.nbrow * this.nbcol;
        Double nbm = percent * this.nbrow * this.nbcol;
        
        int minValue=0,maxValue=0;
        
        for(int i=0;i<this.nbrow;i++){
          for(int j=0;j<this.nbcol;j++){
             minValue=this.pixelMatrix[i][j]<minValue?this.pixelMatrix[i][j]:minValue;
             minValue=this.pixelMatrix[i][j]>minValue?this.pixelMatrix[i][j]:minValue;
            }
        }
        double step=(minValue-minValue)/50;
        double frequences[] = new double[50];
         double centreclasse[] = new double[50];
        //Initialiser à 0;
        double a=0,b=minValue;
       for(int  i=0;i<50;i++){
             frequences[i]=0;
             centreclasse[i]=(a+b)/2;
             a=a+step;
             b=b+step;
        };
        
        double plage=step;
        for (int k = 0; k < 50; k++) {
            for (int i = 0; i < this.nbrow; i++) {
                for (int j = 0; j < this.nbcol; j++) {
                   // minValue = this.pixelMatrix[i][j] < minValue ? this.pixelMatrix[i][j] : minValue;                    
                    if(this.pixelMatrix[i][j]<plage)frequences[i]=frequences[i]++;
                }
            }
            plage+=step;
        }
        
        
        

        IntStream.range(0, 50).parallel().forEach( i->{
             System.out.println("---"+frequences[i]+" ----");
        });
        
       // System.out.println("  NB_row=" + this.pixelMatrix.length+" -- "+this.nbrow+"  NB col="+this.pixelMatrix[0].length+"---"+this.nbcol);
        int number[] = generateRange(nbm.intValue(), 0, this.nbrow * this.nbcol);
        int j = 0;
        for (int i : number) {

            // rand.forEach(i -> {
            // double rowD = i %this.nbcol;
            int row = (int) (i / this.nbcol);
            int col = i - row * this.nbcol;

            this.pixelMatrix[row][col] = -1;
            this.img.setRGB(col, row, new Color(255, 255, 255).getRGB());
        }

    }
    /**
     * Génération du bruit selon un vecteur ??
     * @param vecteur Vecteur de génération du bruit
     * @return ??
     */
    public  int[][] GenerateBruit( int []vecteur) {   
        
        int j=0;
        for(int i: vecteur){
        
       // rand.forEach(i -> {
           // double rowD = i %nbcol;
            int row = (int) (i /nbcol);
            int col = i - row * nbcol;
          
            if(pixelMatrix[row][col]==-1){
                
                //System.out.println(" COl =["+row+" ],["+col+"]");
                 j++;
            }
            pixelMatrix[row][col] = -1; 
            //System.out.println(" NB_J"+i);
        
        }
        return pixelMatrix;
    }
  
    
    /**
     * Génération de bruit en fonction des coordonnées des pixels manquants
     * @param CoodinateL list of position of missing value 
     */
    public void GenerateBruit( ArrayList<double[]> CoodinateL) {      
     
        OriginalImageCopie = new IImage(img);
        // System.out.println(pixelMatrix[0].length+"---"+pixelMatrix.length);
        for(double i[]: CoodinateL){
        
       // rand.forEach(i -> {
           // double rowD = i %nbcol;
           // System.out.println(i[0]+"---"+i[1]);
           if(i[2]==-1){
                int row = ((int)i[0])-1;
                int col = ((int)i[1])-1;             
                this.pixelMatrix[row][col] = -1;  
                this.img.setRGB(col,row , new Color(255, 255, 255).getRGB());   
           }
        }   
   }
    
    /**
     * Sauvegarde l'image dans le fichier
     * @param fileName nom du ficher où sauvegarder l'image
     */
    public void SaveImage(String fileName) {
        // BufferedImage image2 = new BufferedImage(this.nbcol, nbrow, BufferedImage.TYPE_BYTE_GRAY);
        try {
            // retrieve image
            // BufferedImage bi = getMyImage();
            File outputfile = new File(fileName);
            ImageIO.write(this.img, "jpg", outputfile);
        } catch (IOException e) {

        }

    }

    /**
     * Coupe l'image en autres plus petites
     * @param nbpiece nombre de morceaux d'image total
     * @return Un ensemble d'image (ArrayList)
     */
    public ArrayList<IImage> sliceImage(int nbpiece) {

        int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        return sliceImage(nbpiece_per_row, nbpiece_per_row);
    }
    /**
     * Coupe l'image en autres plus petites selon une position
     * @param nbpiece nombre morceaux total
     * @param position Position de la coupe
     * @return Un ensemble d'image (ArrayList)
     */
    public ArrayList<IImage> sliceImage(int nbpiece, Map position) {

        int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        return sliceImage(nbpiece_per_row, nbpiece_per_row,position);
    }
    
    /**
     * Coupe l'image en autres plus petites selon une position et un nombre d'image par lignes et colonnes
     * @param nb_r Nombre d'image par lignes
     * @param nb_c Nombre d'images par colonnes
     * @param position Position de la coupe
     * @return Un ensemble d'image (ArrayList)
     */
    public ArrayList<IImage> sliceImage(int nb_r, int nb_c,Map position) {

        //int nbpiece_per_row = int nb_R;
       // int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        int NB_R = (int) pixelMatrix.length / nb_r;
        int NB_C = pixelMatrix[0].length / nb_c;
        int y = 0, L = 0;
        ArrayList<IImage> Result = new ArrayList();
        for (int t = 0; t < nb_r; t++) {

            int heigth_si = NB_R;

            if (heigth_si * nb_r != pixelMatrix.length && t == 0) {
                heigth_si = heigth_si + (pixelMatrix.length - heigth_si * nb_r);

            }
            int b = 0;

            for (int i = 0; i < nb_c; i++) {

                int width_si = NB_C; //imageMatrix[0].length/nbpiece_per_row;

                if ((width_si * nb_c != pixelMatrix[0].length) && i == 0) {
                    width_si = width_si + (pixelMatrix[0].length - width_si * nb_c);

                }

                //#Result[y]<- matrix(, nrow = NB_R, ncol = NB_C)
                int[][] Moceau = new int[heigth_si][width_si];
                //System.out.println(" NC="+NB_C);
                for (int j = 0; j < heigth_si; j++) {
                    for (int k = 0; k < width_si; k++) {

                        Moceau[j][k] = pixelMatrix[j + L][k + b];
                       // System.out.println(Moceau[j][k]);
                    }
                }
                 
                Result.add(new IImage(Moceau));
                
                int pos[]= {L,b};
                //System.err.println("   L="+L+"   B="+b);
                position.put(y, pos);
                b = b + width_si;      
                //#Result[y]$data<-Moceau

                y++;

            }
            L = L + heigth_si;
        }

        

        return Result;
    }

    /**
     * Coupe l'image en autres plus petites selon un nombre d'image par ligne et colonne
     * @param nb_r Nombre d'image par lignes
     * @param nb_c Nombre d'images par colonnes
     * @return Un ensemble d'image (ArrayList)
     */
    public ArrayList<IImage> sliceImage(int nb_r, int nb_c) {

        //int nbpiece_per_row = int nb_R;
       // int nbpiece_per_row = (int) Math.sqrt(nbpiece);

        int NB_R = (int) pixelMatrix.length / nb_r;
        int NB_C = pixelMatrix[0].length / nb_c;
        int y = 0, L = 0;
        ArrayList<IImage> Result = new ArrayList();
        for (int t = 0; t < nb_r; t++) {

            int heigth_si = NB_R;

            if (heigth_si * nb_r != pixelMatrix.length && t == 0) {
                heigth_si = heigth_si + (pixelMatrix.length - heigth_si * nb_r);

            }
            int b = 0;

            for (int i = 0; i < nb_c; i++) {

                int width_si = NB_C; //imageMatrix[0].length/nbpiece_per_row;

                if ((width_si * nb_c != pixelMatrix[0].length) && i == 0) {
                    width_si = width_si + (pixelMatrix[0].length - width_si * nb_c);

                }

                //#Result[y]<- matrix(, nrow = NB_R, ncol = NB_C)
                int[][] Moceau = new int[heigth_si][width_si];
                //System.out.println(" NC="+NB_C);
                for (int j = 0; j < heigth_si; j++) {
                    for (int k = 0; k < width_si; k++) {

                        Moceau[j][k] = pixelMatrix[j + L][k + b];
                       // System.out.println(Moceau[j][k]);
                    }
                }
                b = b + width_si;        
                Result.add(new IImage(Moceau));
                //#Result[y]$data<-Moceau

                y++;

            }
            L = L + heigth_si;
        }

        

        return Result;
    }

    /**
     * Recoller les morceau de l'image
     * @param liste_morceau Liste des morceaux de l'images
     * @return Une image (IImage)
     */
    public IImage RecollerMorceau(ArrayList<IImage> liste_morceau) {

        int nbpiece_per_row = (int) Math.sqrt(liste_morceau.size());

        int w = 0, h = 0;

        for (int t = 0; t < nbpiece_per_row; t++) {
            h += liste_morceau.get(t).nbcol;// ncol(liste_morceau[[t]])+nco
            
           // System.out.println(" Piece="+t * nbpiece_per_row);
            w += liste_morceau.get(t * nbpiece_per_row).nbrow;
        }
        int imageMatrix[][] = new int[w][h];

        int y = 0, L = 0;

        for (int t = 0; t < nbpiece_per_row; t++) {
            int heigth_si = liste_morceau.get(y).getNbrow();
            int b = 0;
            for (int i = 0; i < nbpiece_per_row; i++) {

                int width_si = liste_morceau.get(y).getNbcol();
                for (int j = 0; j < heigth_si; j++) {
                    for (int k = 0; k < width_si; k++) {
                        imageMatrix[j + L][k + b] = liste_morceau.get(y).getPixelMatrix()[j][k];
                    }
                }

                b = b + width_si;
                y++;
            }
            L = L + heigth_si;

        }

        // this.IImage(imageMatrix);
        IImage image = new IImage(imageMatrix);
        return image;
    }


   
}
