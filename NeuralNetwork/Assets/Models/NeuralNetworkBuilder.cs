public class NeuralNetworkBuilder :
    INeuralNetworkConfigurationStage,
    IInputLayerConfigurationStage,
    IHiddenLayerConfigurationStage
{
    private NetworkConfiguration networkConfiguration = new();

    public static INeuralNetworkConfigurationStage CreateNetwork()
    {
        return new NeuralNetworkBuilder();
    }

    public IInputLayerConfigurationStage WithSettings(double learningRate = 0.02, IErrorFunction errorFunction = null)
    {
        networkConfiguration = new NetworkConfiguration
        {
            LearningRate = learningRate,
            ErrorFunction = errorFunction ?? new PowerDifferenceErrorFunction()
        };
        return this;
    }

    public IHiddenLayerConfigurationStage WithInputLayer(int neurons)
    {
        networkConfiguration.LayerConfigurations.Add(
            new LayerConfiguration { Neurons = neurons }
        );
        return this;
    }

    public IHiddenLayerConfigurationStage WithHiddenLayer(LayerConfiguration layerConfiguration)
    {
        networkConfiguration.LayerConfigurations.Add(layerConfiguration);
        return this;
    }

    public NeuralNetwork WithOutputLayer(LayerConfiguration layerConfiguration)
    {
        networkConfiguration.LayerConfigurations.Add(layerConfiguration);
        return new NeuralNetwork(networkConfiguration);
    }
}

public interface INeuralNetworkConfigurationStage
{
    public IInputLayerConfigurationStage WithSettings(double learningRate = 0.02, IErrorFunction errorFunction = null);
}

public interface IInputLayerConfigurationStage
{
    public IHiddenLayerConfigurationStage WithInputLayer(int neurons);
}

public interface IHiddenLayerConfigurationStage
{
    public IHiddenLayerConfigurationStage WithHiddenLayer(LayerConfiguration layerConfiguration);
    public NeuralNetwork WithOutputLayer(LayerConfiguration layerConfiguration);
}