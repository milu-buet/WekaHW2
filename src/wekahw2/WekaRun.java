/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package wekahw2;

/**
 *
 * @author milu
 */
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.*;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.*;

import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.unsupervised.attribute.NumericToNominal;

import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
 
public class WekaRun {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
	public static Instances make_discrete(Instances inp) {
		Instances discretedata=null;
		try {
		//int cindex = inp.numAttributes() - 1;
		Discretize  filter = new Discretize();
		filter.setInputFormat(inp);
        //String[] options= new String[2];
        //options[0]="-c";
        //options[1]="last";  //range of variables to make numeric
		//filter.setOptions(options);
		//filter.setInputFormat(inp);
		// apply filter
		discretedata = Filter.useFilter(inp, filter);
		} catch(Exception e)
		{System.out.println(e.getMessage());}
		return discretedata;
	}  
	public static Instances make_nominal_class(Instances inp) {
		Instances nominaldata=null;
		try {
		int cindex = inp.numAttributes();
        NumericToNominal convert= new NumericToNominal();
        String[] options= new String[2];
        options[0]="-R";
        options[1]=String.valueOf(cindex);  //range of variables to make numeric
        convert.setOptions(options);
        convert.setInputFormat(inp);

        nominaldata=Filter.useFilter(inp, convert);
		} catch(Exception e)
		{System.out.println(e.getMessage());}
		return nominaldata;
	}  
	public static Instances load_csv(String filename) {
		Instances data=null;
		try {
		BufferedReader datafile = readDataFile(filename);
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(filename));
		data = loader.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		} catch(Exception e)
		{}
		return data;
	}
	public static Instances subsample(Instances inp,int spread_param_val) {
		Instances data=null;
		try {
			SpreadSubsample  filter = new SpreadSubsample();
			filter.setInputFormat(inp);
		    String[] options= new String[2];
		    options[0]="-M";
		    options[1]=String.valueOf(spread_param_val); //how to change the distribution spread of instances, a value of 1 will make it 1:1
			filter.setOptions(options);
			//filter.setInputFormat(inp);
			// apply filter
			data = Filter.useFilter(inp, filter);
		} catch(Exception e)
		{}
		return data;
	}
        
        
        public static void RunTests(Instances data) throws Exception{
           
           Classifier[] cls_s = {new Logistic(), new NaiveBayes(),new IBk(3)};   // k = 3
           String title_str = "Classifier   Ratio   f1-score";
           System.out.println(title_str);
            
           for(Classifier cls: cls_s) {
                 for(int ratio = 10 ;ratio > 0 ; ratio--) {
                        
                        Instances newdata = subsample(data,ratio);
                        Evaluation eval = new Evaluation(newdata);
                        Random rand = new Random(1);  // using seed = 1
                        int folds = 10;
                        eval.crossValidateModel(cls, newdata, folds, rand);
                        
                        //System.out.println(eval.toClassDetailsString());
                        //System.out.println(eval.fMeasure(1));   // P = 1
                        
                        double f1_score = eval.fMeasure(1);  // P = 1
                        String f1_score_str = Double.toString(f1_score);
                        
                        String className = cls.getClass().getSimpleName();
                        
                        String ratio_str = Integer.toString(ratio);
                        
                        String case_output = className + "      " + ratio_str + ":1       " + f1_score_str;
                        System.out.println(case_output);
                  }
           
           }
             
           
        
        }
	
	public static void main(String[] args) throws Exception {
		Instances data = load_csv("imbalanced.csv");
		//Instances data1 = make_nominal_class(data);
		//Instances newdata = make_discrete(data1);
		//subsample the data to make it a 1:1 ratio
//		Instances newdata = subsample(data,1); 
//		Classifier cls = new Logistic();
//		Evaluation eval = new Evaluation(newdata);
//		Random rand = new Random(1);  // using seed = 1
//		int folds = 10;
//		eval.crossValidateModel(cls, newdata, folds, rand);
//		System.out.println(eval.toClassDetailsString());

                RunTests(data);
	}
}