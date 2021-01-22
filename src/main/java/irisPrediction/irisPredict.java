package irisPrediction;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class irisPredict {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		
		String[] classNames = {"Iris-setosa","Iris-versicolor","Iris-virginica"};
		
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));
		
		INDArray predictData = Nd4j.create(new double[][]{
			{5.1,3.5,1.4,0.2},
			{4.9,3.0,1.4,0.2},
			{6.7,3.1,4.4,1.4}
		});
		
		INDArray output = model.output(predictData);
		int[] classes = output.argMax(1).toIntVector();
		System.out.println(output);
		for(int i=0; i<classes.length; i++) {
			System.out.println("Classe : "+ classNames[classes[i]]);
		}
	}

}


