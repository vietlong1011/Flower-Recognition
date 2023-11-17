package org.example;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;

import java.io.File;
import java.util.Scanner;

public class IrisClassifierCSV {

    public static void main(String[] args) {
        try {
            // Load dataset from TXT
            CSVLoader loader = new CSVLoader();
          //  loader.setSource(new File("C:/Flower Recognition/Flower Recognition java/src/main/java/org/example/dataset.txt"));
            loader.setSource(new File("src/main/java/org/example/dataset.txt"));
            Instances data = loader.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            Classifier network  = new MultilayerPerceptron();
            System.out.println("Network : " + network);

            /** hard code **/
            // Build classifier
//            Classifier classifier = new SMO();
//            classifier.buildClassifier(data);
//
//            // Create a new instance for prediction
//            double[] values = {5.1, 3.5, 1.4, 0.2};
//            Instance newInstance = new DenseInstance(1.0, values);
//            newInstance.setDataset(data);
//
//            // Classify the new instance
//            double prediction = classifier.classifyInstance(newInstance);
//            String predictedClass = data.classAttribute().value((int) prediction);

              /**input key**/
            // Build classifier
            Classifier classifier = new SMO();
            classifier.buildClassifier(data);

            // Nhập thông tin từ người dùng
            Scanner scanner = new Scanner(System.in);
            System.out.print("Độ dài đài hoa: ");
            double sepalLength = scanner.nextDouble();

            System.out.print("Độ rộng đài hoa: ");
            double sepalWidth = scanner.nextDouble();

            System.out.print("Độ dài cánh hoa: ");
            double petalLength = scanner.nextDouble();

            System.out.print("Độ rộng cánh hoa: ");
            double petalWidth = scanner.nextDouble();

            // Tạo một instance mới từ thông tin người dùng
            double[] values = {sepalLength, sepalWidth, petalLength, petalWidth};
            Instance newInstance = new DenseInstance(1.0, values);
            newInstance.setDataset(data);

            // Dự đoán lớp của instance mới
            double prediction = classifier.classifyInstance(newInstance);
            String predictedClass = data.classAttribute().value((int) prediction);

            //printf
            System.out.print("\nĐộ dài đài hoa: " + sepalLength);
            System.out.print("\nĐộ rộng đài hoa: " + sepalWidth);
            System.out.print("\nĐộ dài cánh hoa: " + petalLength);
            System.out.print("\nĐộ rộng cánh hoa: " + petalWidth);
            System.out.println("\n\n\tPredicted class: " + predictedClass);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}