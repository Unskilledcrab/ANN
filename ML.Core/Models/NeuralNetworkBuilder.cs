using System.Collections.Generic;
using System;

public class NeuralNetworkBuilder :
    INeuralNetworkConfigurationStage,
    IInputLayerConfigurationStage,
    IHiddenLayerConfigurationStage,
    IBuildNetworksStage
{
    private NetworkConfiguration networkConfiguration = new NetworkConfiguration();

    public static INeuralNetworkConfigurationStage CreateNetwork()
    {
        return new NeuralNetworkBuilder();
    }

    public IInputLayerConfigurationStage WithSettings(Action<NetworkSettings> config = null)
    {
        var networkSettings = new NetworkSettings();
        config?.Invoke(networkSettings);
        networkConfiguration = new NetworkConfiguration
        {
            NetworkSettings = networkSettings
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

    public IHiddenLayerConfigurationStage WithHiddenLayer(Action<LayerConfiguration> config = null)
    {
        var layerConfiguration = new LayerConfiguration();
        config?.Invoke(layerConfiguration);
        networkConfiguration.LayerConfigurations.Add(layerConfiguration);
        return this;
    }

    public IBuildNetworksStage WithOutputLayer(Action<LayerConfiguration> config = null)
    {
        var layerConfiguration = new LayerConfiguration();
        config?.Invoke(layerConfiguration);
        networkConfiguration.LayerConfigurations.Add(layerConfiguration);
        return this;
    }

    public NeuralNetwork Build()
    {
        return new NeuralNetwork(networkConfiguration);
    }

    public List<NeuralNetwork> Build(int networkCount)
    {
        List<NeuralNetwork> networks = new List<NeuralNetwork>();
        for (int i = 0; i < networkCount; i++)
        {
            networks.Add(new NeuralNetwork(networkConfiguration));
        }
        return networks;
    }
}

public interface INeuralNetworkConfigurationStage
{
    public IInputLayerConfigurationStage WithSettings(Action<NetworkSettings> config = null);
}

public interface IInputLayerConfigurationStage
{
    public IHiddenLayerConfigurationStage WithInputLayer(int neurons);
}

public interface IHiddenLayerConfigurationStage
{
    public IHiddenLayerConfigurationStage WithHiddenLayer(Action<LayerConfiguration> config = null);
    public IBuildNetworksStage WithOutputLayer(Action<LayerConfiguration> config = null);
}

public interface IBuildNetworksStage
{
    public NeuralNetwork Build();
    public List<NeuralNetwork> Build(int networkCount);
}