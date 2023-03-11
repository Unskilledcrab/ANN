using System.Collections.Generic;

public class NetworkConfiguration
{
    public double LearningRate { get; set; } = 0.02;
    public IErrorFunction ErrorFunction { get; set; } = new PowerDifferenceErrorFunction();
    public List<LayerConfiguration> LayerConfigurations { get; set; } = new();
}   
