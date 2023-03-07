public class NetworkConfiguration
{
    public double LearningRate { get; set; }
    public IErrorFunction ErrorFunction { get; set; } = new PowerDifferenceErrorFunction();
    public List<LayerConfiguration> LayerConfigurations { get; set; } = new();
}   
