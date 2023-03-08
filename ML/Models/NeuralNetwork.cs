public class NeuralNetwork
{
    private double outputMax;
    private double outputMin;

    public List<NeuralLayer> Layers { get; set; } = new();
    public NeuralLayer InputLayer { get; private set; }
    public NeuralLayer OutputLayer { get; private set; }
    public List<double> Errors { get; private set; } = new();
    public IErrorFunction ErrorFunction { get; private set; }
    public double LearningRate { get; }
    public List<NetworkStats> NetworkStats { get; set; } = new();
    public NeuralNetwork(NetworkConfiguration networkConfiguration)
    {
        ErrorFunction = networkConfiguration.ErrorFunction;
        LearningRate = networkConfiguration.LearningRate;
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

    public List<TrainingSet> NormalizeTrainingData(List<TrainingSet> data)
    {
        var inputMax = data.SelectMany(d => d.Inputs).Max();
        var inputMin = data.SelectMany(d => d.Inputs).Min();
        outputMax = data.SelectMany(d => d.ExpectedOutputs).Max();
        outputMin = data.SelectMany(d => d.ExpectedOutputs).Min();

        for (int i = 0; i < data.Count; i++)
        {
            var set = data[i];
            for (int j = 0; j < set.Inputs.Count; j++)
            {
                var value = set.Inputs[j];
                set.Inputs[j] = Normalize(value, inputMin, inputMax) - 0.5;
            }
            for (int j = 0; j < set.ExpectedOutputs.Count; j++)
            {
                var value = set.ExpectedOutputs[j];
                set.ExpectedOutputs[j] = Normalize(value, outputMin, outputMax) - 0.5;
            }
        }
        return data;
    }

    public double Normalize(double input, double min, double max)
    {
        return (input - min) / (max - min);
    }

    public double DeNormalize(double input)
    {
        return (input * (outputMax - outputMin)) + outputMin;
    }

    public void Train(List<TrainingSet> dataSets, int epochs)
    {
        //data = NormalizeTrainingData(data);
        Console.WriteLine("Starting Training");
        for (int i = 0; i < epochs; i++)
        {
            var averageError = SplitAndTrain(dataSets);
            var stats = new NetworkStats
            {
                Epoch = i + 1,
                Error = averageError
            };
            
            Console.WriteLine($"Epoch: {stats.Epoch.ToString("0000")} \t Error: {stats.Error}");
            NetworkStats.Add(stats);
        }
        Console.WriteLine();
    }

    private double SplitAndTrain(List<TrainingSet> dataSets)
    {
        var validationCount = (int)Math.Floor(dataSets.Count * 0.2);
        var trainingCount = dataSets.Count - validationCount;

        foreach (var set in dataSets.Take(trainingCount))
        {
            Train(set);
        }

        var totalAverageErrors = new List<double>();
        foreach (var set in dataSets.Skip(trainingCount).Take(validationCount))
        {
            SetInputs(set.Inputs);
            CalculateError(set.ExpectedOutputs);
            totalAverageErrors.Add(Errors.Average());
        }
        var networkAverageError = totalAverageErrors.Average();
        return networkAverageError;
    }

    public void Train(TrainingSet set)
    {
        SetInputs(set.Inputs);
        UpdateLayers(set.ExpectedOutputs);
    }

    private void UpdateLayers(List<double> expectedOutputs)
    {
        UpdateOutputLayer(expectedOutputs);
        UpdateHiddenLayers();
    }

    public void UpdateHiddenLayers()
    {
        if (Layers.Count <= 2)
        {
            return; // There are no hidden layers to train
        }

        // Skip the first and last layer and train in reverse order
        for (int i = Layers.Count - 1; i > 0; i--)
        {
            Layers[i].TrainHiddenNeurons(LearningRate);
        }
    }

    private void UpdateOutputLayer(List<double> expectedOutputs)
    {
        OutputLayer.TrainOutputNeurons(expectedOutputs, ErrorFunction, LearningRate);
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
        SetDirty(true);
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

    private void SetDirty(bool isDirty)
    {
        foreach (var layer in Layers)
        {
            layer.SetDirty(isDirty);
        }
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
