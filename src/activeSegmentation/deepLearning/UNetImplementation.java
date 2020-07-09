package activeSegmentation.deepLearning;

import activeSegmentation.IDeepLearning;
import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;


public class UNetImplementation implements IDeepLearning {
    private static final int seed = 1234;
    private WeightInit weightInit = WeightInit.RELU;
    protected static Random rng = new Random(seed);
    protected static int epochs = 1;
    private static int batchSize = 1;

    private static int width = 512;
    private static int height = 512;
    private static int channels = 3;
    public static final String dataPath = "/home/jstachera/Documents/data";



    public void importData() throws IOException {
        File trainData = new File(dataPath + "/train/image");
        File testData = new File(dataPath + "/test/image");

        LabelGenerator labelMakerTrain = new LabelGenerator(dataPath + "/train");
        LabelGenerator labelMakerTest = new LabelGenerator(dataPath + "/test");

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, rng);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, rng);


        ImageRecordReader rrTrain = new ImageRecordReader(height, width, channels, labelMakerTrain);
        rrTrain.initialize(train, null);

        ImageRecordReader rrTest = new ImageRecordReader(height, width, channels, labelMakerTest);
        rrTest.initialize(test, null);

        int labelIndex = 1;
        DataSetIterator dataTrainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, labelIndex, labelIndex, true);
        DataSetIterator dataTestIter = new RecordReaderDataSetIterator(rrTest, 1, labelIndex, labelIndex, true);
    }
    public ComputationGraph buildModel(DataSetIterator dataTrainIter, DataSetIterator dataTestIter) throws IOException {
        DL4JResources.setBaseDownloadURL("https://dl4jdata.blob.core.windows.net/");
        ZooModel zooModel = UNet.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained(PretrainedType.SEGMENT);
        //System.out.println(pretrainedNet.summary());
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0, 1);
        scaler.fitLabel(true);
        scaler.fit(dataTrainIter);
        dataTrainIter.setPreProcessor(scaler);
        scaler.fit(dataTestIter);
        dataTestIter.setPreProcessor(scaler);
        System.out.println(pretrainedNet.summary());

        ComputationGraph unetTransfer = new TransferLearning.GraphBuilder(pretrainedNet)
                .setFeatureExtractor("conv2d_23")
                .removeVertexKeepConnections("activation_23")
                .addLayer("activation_23",
                        new CnnLossLayer.Builder(LossFunctions.LossFunction.XENT)
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.SIGMOID).build(), "conv2d_23")
                .build();

        System.out.println(unetTransfer.summary());

        unetTransfer.init();
        System.out.println(unetTransfer.summary());
        return unetTransfer;
    }
    public void train(ComputationGraph unetTransfer, DataSetIterator dataTrainIter) {
            unetTransfer.fit(dataTrainIter, epochs);

    }
    public double[] evaluate(DataSetIterator dataTestIter, NormalizerMinMaxScaler scaler, ComputationGraph unetTransfer){

            DataSet t = dataTestIter.next();
            scaler.revert(t);
            INDArray[] predicted = unetTransfer.output(t.getFeatures());
            INDArray input = t.getFeatures();
            INDArray pred = predicted[0].reshape(new int[]{512, 512});
            Evaluation eval = new Evaluation();

            eval.eval(pred.dup().reshape(512 * 512, 1), t.getLabels().dup().reshape(512 * 512, 1));
            System.out.println(eval.stats());
            DataBuffer dataBuffer = pred.data();
            double[] classificationResult = dataBuffer.asDouble();
            ImageProcessor classifiedSliceProcessor = new FloatProcessor(512, 512, classificationResult);
            int j = 0;
            //segmented image instance
            ImagePlus classifiedImage = new ImagePlus("pred" + j, classifiedSliceProcessor);
            IJ.save(classifiedImage, dataPath + "/predict/pred-" + j + ".png");
            j++;
            return classificationResult;
        }
        public String toString(ComputationGraph uNet){
            return uNet.summary();
        }
    }