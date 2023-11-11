package org.example;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.converters.CSVLoader;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;


import java.io.File;
import java.util.Scanner;

public class IrisNeuralNetwork {

    public static void main(String[] args) {
        try {
            // Load dataset from CSV
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("C:/testAL/src/main/java/org/example/dataset.txt"));
            Instances data = loader.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Build neural network classifier
            Classifier classifier = new MultilayerPerceptron();
            classifier.buildClassifier(data);
//
//            // Nhập thông tin từ người dùng
//            Scanner scanner = new Scanner(System.in);
//            System.out.print("Độ dài đài hoa: ");
//            double sepalLength = 5.1;//scanner.nextDouble();
//
//            System.out.print("Độ rộng đài hoa: ");
//            double sepalWidth = 3.5;//scanner.nextDouble();
//
//            System.out.print("Độ dài cánh hoa: ");
//            double petalLength = 1.4;//scanner.nextDouble();
//
//            System.out.print("Độ rộng cánh hoa: ");
//            double petalWidth =  0.2;  //scanner.nextDouble();

            // Tạo một DenseInstance mới từ thông tin người dùng
           // DenseInstance newInstance = new DenseInstance(1.0, new double[]{sepalLength, sepalWidth, petalLength, petalWidth});
            double[] values = {5.1, 3.5, 1.4, 0.2};
            DenseInstance newInstance = new DenseInstance(1.0, values);
            newInstance.setDataset(data);

            // Dự đoán lớp của instance mới
            double[] prediction = classifier.distributionForInstance(newInstance);

            // Hiển thị kết quả dự đoán
            System.out.println("Predicted class distribution: ");
            for (int i = 0; i < prediction.length; i++) {
                System.out.println(data.classAttribute().value(i) + ": " + prediction[i]);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


