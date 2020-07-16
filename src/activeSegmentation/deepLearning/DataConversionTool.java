package activeSegmentation.deepLearning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;

public class DataConversionTool {

    /**
     * Converts a set of training instances to a DataSet. Assumes that the instances have been
     * suitably preprocessed - i.e. missing values replaced and nominals converted to binary/numeric.
     * Also assumes that the class index has been set
     *
     * @param insts the instances to convert
     * @return a DataSet
     */
    public static DataSet instancesToDataSet(Instances insts) {
        INDArray data = Nd4j.zeros(insts.numInstances(), insts.numAttributes() - 1);
        INDArray outcomes = Nd4j.zeros(insts.numInstances(), insts.numClasses());

        for (int i = 0; i < insts.numInstances(); i++) {
            double[] independent = new double[insts.numAttributes() - 1];
            double[] dependent = new double[insts.numClasses()];
            Instance current = insts.instance(i);
            for (int j = 0; j < current.numValues(); j++) {
                int index = current.index(j);
                double value = current.valueSparse(j);

                if (index < insts.classIndex()) {
                    independent[index] = value;
                } else if (index > insts.classIndex()) {
                    // Shift by -1, since the class is left out from the feature matrix and put into a separate
                    // outcomes matrix
                    independent[index - 1] = value;
                }
            }

            // Set class values
            if (insts.numClasses() > 1) { // Classification
                final int oneHotIdx = (int) current.classValue();
                dependent[oneHotIdx] = 1.0;
            } else { // Regression (currently only single class)
                dependent[0] = current.classValue();
            }

            INDArray row = Nd4j.create(independent);
            data.putRow(i, row);
            outcomes.putRow(i, Nd4j.create(dependent));
        }
        return new DataSet(data, outcomes);
    }

    /**
     * Converts a set of training instances to a DataSet prepared for the convolution operation using
     * the height, width and number of channels
     *
     * @param height image height
     * @param width image width
     * @param channels number of image channels
     * @param insts the instances to convert
     * @return a DataSet
     */
    public static DataSet instancesToConvDataSet(
            Instances insts, int height, int width, int channels) {
        DataSet ds = instancesToDataSet(insts);
        INDArray data = Nd4j.zeros(insts.numInstances(), channels, width, height);
        ds.getFeatures();

        for (int i = 0; i < insts.numInstances(); i++) {
            INDArray row = ds.getFeatures().getRow(i);
            row = row.reshape(1, channels, height, width);
            data.putRow(i, row);
        }

        return new DataSet(data, ds.getLabels());
    }

    private int loadLabels(String directory){
        File folder = new File(directory);
        File[] labels = folder.listFiles();

        for (File file : labels) {
            if (file.isFile()) {

            }
        }
        return -1;
    }
}
