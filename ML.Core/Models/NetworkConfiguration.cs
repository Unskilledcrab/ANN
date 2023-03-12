using System.Collections.Generic;

public class NetworkConfiguration
{
    public NetworkSettings NetworkSettings { get; set; } = new NetworkSettings();
    public List<LayerConfiguration> LayerConfigurations { get; set; } = new List<LayerConfiguration>();
}   
