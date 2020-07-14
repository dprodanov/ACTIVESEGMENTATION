package activeSegmentation;

import org.nd4j.linalg.dataset.api.DataSet;
import weka.core.Instances;

public interface IDeepDataSet {

    /**
     * Converts a set of training instances to a DataSet. Assumes that the instances have been
     * suitably preprocessed - i.e. missing values replaced and nominals converted to binary/numeric.
     * Also assumes that the class index has been set
     *
     * @param insts the instances to convert
     * @return a DataSet
     */
    public DataSet instancesToDataSet(Instances insts);
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
    public DataSet instancesToConvDataSet(
            Instances insts, int height, int width, int channels);
    }
