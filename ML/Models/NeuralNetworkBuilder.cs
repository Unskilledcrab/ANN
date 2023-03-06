public class NeuralNetworkBuilder
{
    public List<LayerConfiguration> LayerConfigurations { get; set; } = new();
    public NeuralNetwork Build()
    {
        if (LayerConfigurations.Count < 2)
        {
            throw new Exception($"{nameof(LayerConfigurations)} must have at least two layers configured (input & output)");
        }
        return new NeuralNetwork(LayerConfigurations);
    }
}
