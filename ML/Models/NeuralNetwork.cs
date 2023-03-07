public class NeuralNetwork
{
    public List<NeuralLayer> Layers { get; set; } = new();
    public NeuralLayer InputLayer { get; private set; }
    public NeuralLayer OutputLayer { get; private set; }
    public List<double> Errors { get; private set; } = new();
    public IErrorFunction ErrorFunction { get; private set; }
    public List<NetworkStats> NetworkStats { get; set; } = new();
    public NeuralNetwork(NetworkConfiguration networkConfiguration)
    {
        ErrorFunction = networkConfiguration.ErrorFunction;
        SetupLayers(networkConfiguration.LayerConfigurations);
    }

    private void SetupLayers(List<LayerConfiguration> configurations)
    {
        if (configurations.Count < 2)
        {
            throw new ArgumentException("You must provide at least two layer configurations (input & output)", nameof(configurations));
        }

        for (int i = 0; i < configurations.Count; i++)
        {
            if (i == 0)
            {
                CreateInputLayer(configurations[i]);
            }
            else if (i == configurations.Count - 1)
            {
                CreateOutputLayer(configurations[i]);
            }
            else
            {
                CreateHiddenLayer(configurations[i]);
            }
        }
    }

    public List<double> Predict(List<double> inputs)
    {
        SetInputs(inputs);
        return GetOutput();
    }

    public void Train(List<TrainingData> data, int epochs)
    {
        Console.WriteLine("Starting Training");
        for (int i = 0; i < epochs; i++)
        {
            var averageError = SplitAndTrain(data);
            var stats = new NetworkStats
            {
                Epoch = i + 1,
                Error = averageError
            };
            NetworkStats.Add(stats);
        }
        Console.WriteLine();
    }

    private double SplitAndTrain(List<TrainingData> data)
    {
        var validationCount = (int)Math.Floor(data.Count * 0.2);
        var trainingCount = data.Count - validationCount;

        foreach (var set in data.Take(trainingCount))
        {
            Train(set);
        }

        var totalAverageErrors = new List<double>();
        foreach (var set in data.Skip(trainingCount).Take(validationCount))
        {
            SetInputs(set.Inputs);
            CalculateError(set.ExpectedOutputs);
            totalAverageErrors.Add(Errors.Average());
        }
        var networkAverageError = totalAverageErrors.Average();
        return networkAverageError;
    }

    public void Train(TrainingData data)
    {
        SetInputs(data.Inputs);
        CalculateError(data.ExpectedOutputs);
        UpdateLayers(data.ExpectedOutputs);
    }

    private void UpdateLayers(List<double> expectedOutputs)
    {
        Errors.Clear();
        var actualOutputs = GetOutput();
        for (int i = 0; i < expectedOutputs.Count; i++)
        {            
            var error = ErrorFunction.CalculateError(actualOutputs[i], expectedOutputs[i]);
            Errors.Add(error);
        }
    }

    public void CalculateError(List<double> expectedOutputs)
    {
        Errors.Clear();
        var actualOutputs = GetOutput();
        for (int i = 0; i < expectedOutputs.Count; i++)
        {            
            var error = ErrorFunction.CalculateError(actualOutputs[i], expectedOutputs[i]);
            Errors.Add(error);
        }
    }

    public void SetInputs(IEnumerable<double> inputs)
    {
        var index = 0;
        foreach (var input in inputs)
        {
            InputLayer.Neurons[index].Value = input;
            index++;
        }
    }

    public List<double> GetOutput()
    {
        var outputs = new List<double>();
        foreach (var outputNeuron in OutputLayer.Neurons)
        {
            outputs.Add(outputNeuron.CalculateOutput());
        }
        return outputs;
    }

    private void CreateInputLayer(LayerConfiguration configuration)
    {
        var inputLayer = new NeuralLayer(configuration.Neurons, configuration.InputFunction, configuration.ActivationFunction);
        Layers.Add(inputLayer);
        InputLayer = inputLayer;
    }

    private void CreateHiddenLayer(LayerConfiguration configuration)
    {
        var hiddenLayer = new NeuralLayer(configuration.Neurons, configuration.InputFunction, configuration.ActivationFunction);
        hiddenLayer.ConnectPreviousLayer(Layers.Last());
        Layers.Add(hiddenLayer);
    }

    private void CreateOutputLayer(LayerConfiguration configuration)
    {
        var outputLayer = new NeuralLayer(configuration.Neurons, configuration.InputFunction, configuration.ActivationFunction);
        outputLayer.ConnectPreviousLayer(Layers.Last());
        Layers.Add(outputLayer);
        OutputLayer = outputLayer;
    }
}
