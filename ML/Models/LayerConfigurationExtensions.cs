
public static class LayerConfigurationExtensions
{
    private static readonly IInputFunction defaultHiddenInputFunction = new WeightedSumFunction();
    private static readonly IActivationFunction defaultHiddenActivationFunction = new RectifiedActivationFunction();
    private static readonly IInputFunction defaultOutputInputFunction = new WeightedSumFunction();
    private static readonly IActivationFunction defaultOutputActivationFunction = new RectifiedActivationFunction();

    public static List<LayerConfiguration> WithInputs(this List<LayerConfiguration> configurations, int neurons)
    {
        var inputConfiguration = new LayerConfiguration { Neurons = neurons };
        configurations.Add(inputConfiguration);
        return configurations;
    }

    public static List<LayerConfiguration> WithHiddenLayer(
        this List<LayerConfiguration> configurations, 
        int neurons,
        IInputFunction? inputFunction = null,
        IActivationFunction? activationFunction = null)
    {
        inputFunction ??= defaultHiddenInputFunction;
        activationFunction ??= defaultHiddenActivationFunction;
        var inputConfiguration = new LayerConfiguration 
        { 
            Neurons = neurons, 
            InputFunction = inputFunction,
            ActivationFunction = activationFunction 
        };
        configurations.Add(inputConfiguration);
        return configurations;
    }

    public static List<LayerConfiguration> WithOutputLayer(
        this List<LayerConfiguration> configurations, 
        int neurons,
        IInputFunction? inputFunction = null,
        IActivationFunction? activationFunction = null)
    {
        inputFunction ??= defaultOutputInputFunction;
        activationFunction ??= defaultOutputActivationFunction;
        var inputConfiguration = new LayerConfiguration 
        { 
            Neurons = neurons, 
            InputFunction = inputFunction,
            ActivationFunction = activationFunction 
        };
        configurations.Add(inputConfiguration);
        return configurations;
    }
}