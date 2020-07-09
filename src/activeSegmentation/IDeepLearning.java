package activeSegmentation;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

import java.io.IOException;

public interface IDeepLearning {
    public void importData() throws IOException;

    public void train(ComputationGraph unetTransfer, DataSetIterator dataTrainIter);

    public ComputationGraph buildModel(DataSetIterator dataTrainIter, DataSetIterator dataTestIter) throws IOException;

    public double[] evaluate(DataSetIterator dataTestIter, NormalizerMinMaxScaler scaler, ComputationGraph unetTransfer);

}
