package activeSegmentation.deepLearning;

import activeSegmentation.ASCommon;
import activeSegmentation.prj.ProjectInfo;
import ij.IJ;
import ij.ImagePlus;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
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


public class UNetImplementation {
    private static final int seed = 1234;
    private WeightInit weightInit = WeightInit.RELU;
    protected static Random rng = new Random(seed);
    protected static int epochs = 1;
    private static int batchSize = 1;

    private static int width = 512;
    private static int height = 512;
    private static int channels = 3;
    private ProjectInfo projectInfo;
    public String dataPath;

    public UNetImplementation(ProjectInfo projectInfo) {
        this.projectInfo = projectInfo;
    }

    public void importData(double proportion) throws IOException {
        dataPath = projectInfo.getProjectDirectory().get(ASCommon.DEEPLEARNINGDIR);
        LabelGenerator labelMaker = new LabelGenerator(dataPath + "labels");
        File mainPath = new File(dataPath+"images");
        FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        int numExamples = Math.toIntExact(fileSplit.length());
        int numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length;
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, 1);

        InputSplit[] inputSplit = fileSplit.sample(pathFilter, proportion, 1 - proportion);
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];



        ImageRecordReader rrTrain = new ImageRecordReader(height, width, channels, labelMaker);
        rrTrain.initialize(trainData, null);

        ImageRecordReader rrTest = new ImageRecordReader(height, width, channels, labelMaker);
        rrTest.initialize(testData, null);

        int labelIndex = 1;
        DataSetIterator dataTrainIter = new RecordReaderDataSetIterator(rrTrain, batchSize, labelIndex, labelIndex, true);
        DataSetIterator dataTestIter = new RecordReaderDataSetIterator(rrTest, 1, labelIndex, labelIndex, true);

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


        unetTransfer.init();
        System.out.println(unetTransfer.summary());
            unetTransfer.fit(dataTrainIter, epochs);

            DataSet t = dataTestIter.next();
            scaler.revert(t);
            INDArray[] predicted = unetTransfer.output(t.getFeatures());
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
            IJ.save(classifiedImage, dataPath + "/predictions/"+ j + ".png");
            j++;
        }
        public String toString(ComputationGraph uNet){
            return uNet.summary();
        }
    }