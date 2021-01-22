package irisTraining;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class irisTrain {

	public static void main(String[] args) throws IOException, InterruptedException {
		// TODO Auto-generated method stub

		double learninRate = 0.001;
		int numInputs = 4;
		int numHidden = 10;
		int numOutputs = 3;
		int classIndex = 4;
		int batchSize = 1;
		int outputSize = 3;
		int numEpochs = 100;
		
		MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
				.seed(123)
				.updater(new Adam(learninRate))
				.list()
					.layer(0, new DenseLayer.Builder()
								.nIn(numInputs)
								.nOut(numHidden)
								.activation(Activation.SIGMOID)
								.build()
					)
					.layer(1, new OutputLayer.Builder()
								.nIn(numHidden)
								.nOut(numOutputs)
								.activation(Activation.SOFTMAX)
								.lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
								.build()
					)
					.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(config);
		model.init();
		
		UIServer uiServer = UIServer.getInstance();
		InMemoryStatsStorage inMemory = new InMemoryStatsStorage();
		uiServer.attach(inMemory);
		
		model.setListeners(new StatsListener(inMemory));
		
		//System.out.println("Configuration :");
		//System.out.println(config.toString());
		
		File fileTrain = new ClassPathResource("iris-train.csv").getFile();
		RecordReader recordReaderTrain = new CSVRecordReader();
		recordReaderTrain.initialize(new FileSplit(fileTrain));
		DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain,
																			   batchSize,
																			   classIndex,
																			   outputSize);
		
		while(dataSetIteratorTrain.hasNext()) {
			DataSet dataSet = dataSetIteratorTrain.next();
			System.out.println(dataSet.getFeatures());
			System.out.println(dataSet.getLabels().toString());
		}
		
		for(int i=0; i<numEpochs; i++) {
			model.fit(dataSetIteratorTrain);
		}
		
		
		File fileTest = new ClassPathResource("iris-test.csv").getFile();
		RecordReader recordReaderTest = new CSVRecordReader();
		recordReaderTest.initialize(new FileSplit(fileTest));
		DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest,
																			  batchSize,
																			  classIndex,
																			  numOutputs);
		Evaluation evaluation = new Evaluation(numOutputs);
		
		while(dataSetIteratorTest.hasNext()) {
			DataSet dataSet = dataSetIteratorTest.next();
			INDArray features = dataSet.getFeatures();
			INDArray labels = dataSet.getLabels();
			INDArray predict = model.output(features);
			evaluation.eval(labels, predict);
		}
		
		System.out.println("-------------------------- Evaluation --------------------------");
		System.out.println(evaluation.stats());
		
		
		ModelSerializer.writeModel(model, "irisModel.zip", true);
	}

}
