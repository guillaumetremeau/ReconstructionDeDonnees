/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import Inpainting.ImageInpaint;
import java.io.IOException;
import java.io.Writer;
import java.util.List;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author koueya
 */
public class CSVUtils {

    private static final char DEFAULT_SEPARATOR = ';';

    public static void writeLine(Writer w, List<String> values) throws IOException {
        writeLine(w, values, DEFAULT_SEPARATOR, ' ');
    }

    public static void writeLine(Writer w, int[] values) throws IOException {
        writeLine(w, values, DEFAULT_SEPARATOR, ' ');
    }

    public static void writeLine(Writer w, double[] values) throws IOException {
        writeLine(w, values, DEFAULT_SEPARATOR, ' ');
    }

    public static void writeLine(Writer w, List<String> values, char separators) throws IOException {
        writeLine(w, values, separators, ' ');
    }

    public static void writeLine(Writer w, double[] values, char separators) throws IOException {
        writeLine(w, values, separators, ' ');
    }

    public static void writeLine(Writer w, int[] values, char separators) throws IOException {
        writeLine(w, values, separators, ' ');
    }

    //https://tools.ietf.org/html/rfc4180
    private static String followCVSformat(String value) {

        String result = value;
        if (result.contains("\"")) {
            result = result.replace("\"", "\"\"");
        }
        return result;

    }

    public static void writeLine(Writer w, List<String> values, char separators, char customQuote) throws IOException {

        boolean first = true;

        //default customQuote is empty
        if (separators == ' ') {
            separators = DEFAULT_SEPARATOR;
        }

        StringBuilder sb = new StringBuilder();
        for (String value : values) {
            if (!first) {
                sb.append(separators);
            }
            if (customQuote == ' ') {
                sb.append(followCVSformat(value));
            } else {
                sb.append(customQuote).append(followCVSformat(value)).append(customQuote);
            }

            first = false;
        }
        sb.append("\n");
        w.append(sb.toString());

    }

    public static void writeLine(Writer w, double[] values, char separators, char customQuote) throws IOException {

        boolean first = true;

        //default customQuote is empty
        if (separators == ' ') {
            separators = DEFAULT_SEPARATOR;
        }

        StringBuilder sb = new StringBuilder();
        for (double value : values) {

            String val = value != -1 ? String.valueOf(value) : "NA";
            if (!first) {
                sb.append(separators);
            }
            if (customQuote == ' ') {

                sb.append(followCVSformat(val));

            } else {
                sb.append(customQuote).append(followCVSformat(val)).append(customQuote);
            }

            first = false;
        }
        sb.append("\n");
        w.append(sb.toString());

    }

    public static void writeLine(Writer w, int[] values, char separators, char customQuote) throws IOException {

        boolean first = true;

        //default customQuote is empty
        if (separators == ' ') {
            separators = DEFAULT_SEPARATOR;
        }

        StringBuilder sb = new StringBuilder();
        for (double value : values) {

            String val = value != -1 ? String.valueOf(value) : "NA";
            if (!first) {
                sb.append(separators);
            }
            if (customQuote == ' ') {

                sb.append(followCVSformat(val));

            } else {
                sb.append(customQuote).append(followCVSformat(val)).append(customQuote);
            }

            first = false;
        }
        sb.append("\n");
        w.append(sb.toString());
    }

    public static void toArff(String file, String path) throws IOException {

        String csvfile_name = path + "/" + file + ".csv";

        String arfffile_name = path + "/" + file + ".arff";

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csvfile_name));
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arfffile_name));
        saver.setDestination(new File(arfffile_name));
        saver.writeBatch();

    }

     public static void toArff(String file) throws IOException {
         toArff(file, "data");     
    }
    
     public static ArrayList<String[]> CSVReader(String File) throws FileNotFoundException{
         
    ArrayList<String[]> elements= new ArrayList();   
     Scanner scanner = new Scanner(new File( File));
        scanner.useDelimiter("\\r\\n");
        while(scanner.hasNext()){
          String buf[]= scanner.next().split(""+DEFAULT_SEPARATOR+"");
          elements.add(buf);      
        }
        scanner.close();
        return elements;
     }
         public static void writecsv(double dataFrame[][], String filname) throws IOException {
        //int image2[][] = new int[nbrow][nbcol];
        //System.err.println("_____" + dataFrame.length);
        FileWriter writer = new FileWriter(filname,true);
        int k = 0;
        CSVUtils.writeLine(writer, Arrays.asList("x", "y", "target", "m1", "m2", "m3", "m1/m2", "m1/m3", "w1", "w2", "w3", "w1/w2", "w1/w3", "sd1", "sd2", "sd3"), ',');
        /*IntStream.range(0, dataFrame.length).forEach(new IntConsumer() {
            @Override
            public static void accept(int i) {
                try {
                    CSVUtils.writeLine(writer,dataFrame[i], ';');
                   System.err.println("Value of i:"+i);
                } catch (IOException ex) {
                    Logger.getLogger(LoadImageApp.class.getName()).log(Level.SEVERE, null, ex);
                }}
        });
         */
        for (double[] data : dataFrame) {

            try {
                CSVUtils.writeLine(writer, data, ',');
                // System.err.println("Value of i:"+i);
            } catch (IOException ex) {
                Logger.getLogger(ImageInpaint.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        writer.close();
    }
    
}
