/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import static javafx.scene.input.KeyCode.T;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

/**
 *
 * @author koueya
 */
public class MyUtilsForWekaInstanceHelper {

    public static int classIndex=2;
    private static String[] attributnames = {"x", "y", "target", "m1", "m2", "m3", "m1/m2", "m2/m3", "w1", "w2", "w3", "w1/w2", "w1/w3", "sd1", "sd2", "sd3"};

    public static Instances getInstances(double[][] data) {
        ArrayList attributeList;
        Instances instances;
        attributeList = new ArrayList();
        for (String attributname : attributnames) {

            attributeList.add(new Attribute(attributname));
        }
        instances = new Instances("MyRelation", attributeList, 0);

        for (double[] oneData : data) {

            instances.add(new DenseInstance(1.0, oneData));
        }
        instances.setClassIndex(classIndex);

        return instances;
    }

}
