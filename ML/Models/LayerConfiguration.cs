public class LayerConfiguration
{
    public int Neurons { get; set; }
    public IInputFunction InputFunction { get; set; } = new WeightedSumFunction();
    public IActivationFunction ActivationFunction { get; set; } = new RectifiedActivationFunction();
}