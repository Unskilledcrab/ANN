public class NetworkConfiguration
{
    public int InputNeurons { get; set; }
    public int OutputNeurons { get; set; }
    public IInputFunction InputFunction { get; set; }
    public IActivationFunction ActivationFunction { get; set; }
    public List<LayerConfiguration> HiddenLayerConfigurations { get; set; } = new();
}   
