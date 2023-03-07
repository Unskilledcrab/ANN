public class NeuralNetworkBuilder
{
    public List<LayerConfiguration> LayerConfigurations { get; set; } = new();
    public NeuralNetwork Build()
    {
        if (LayerConfigurations.Count < 2)
        {
            throw new Exception($"{nameof(LayerConfigurations)} must have at least two layers configured (input & output)");
        }
        var networkConfiguration = new NetworkConfiguration
        {
            LayerConfigurations = LayerConfigurations
        };
        return new NeuralNetwork(networkConfiguration);
    }
}
