public class NeuralNetwork
{
    public List<NeuralLayer> Layers { get; private set; } = new();
    public NeuralLayer InputLayer => Layers[0];
    public NeuralLayer OutputLayer => Layers[Layers.Count - 1];
    public List<double> Errors { get; private set; } = new();
    public NetworkSettings NetworkSettings { get; private set; } = new();
    public List<NetworkStats> NetworkStats { get; private set; } = new();

    private double LearningRate => NetworkSettings.LearningRate;
    private IErrorFunction ErrorFunction => NetworkSettings.ErrorFunction;
    public NeuralNetwork(NetworkConfiguration networkConfiguration)
    {
        NetworkSettings = networkConfiguration.NetworkSettings;
        SetupLayers(networkConfiguration.LayerConfigurations);
    }

    private void SetupLayers(List<LayerConfiguration> configurations)
    {
        if (configurations.Count < 2)
        {
            throw new ArgumentException("You must provide at least two layer configurations (input & output)", nameof(configurations));
        }

        foreach (var configuration in configurations)
        {
            AddLayer(configuration);
        }
    }

    public List<double> Predict(List<double> inputs)
    {
        //SetInputs(inputs.Select(i => Normalize(i)));
        SetInputs(inputs);
        return GetOutput();
    }

    public List<TrainingSet> NormalizeTrainingData(List<TrainingSet> data)
    {
        for (int i = 0; i < data.Count; i++)
        {
            var set = data[i];
            for (int j = 0; j < set.Inputs.Count; j++)
            {
                var value = set.Inputs[j];
                set.Inputs[j] = Normalize(value);
            }
            for (int j = 0; j < set.ExpectedOutputs.Count; j++)
            {
                var value = set.ExpectedOutputs[j];
                set.ExpectedOutputs[j] = Normalize(value);
            }
        }
        return data;
    }

    public double Normalize(double input)
    {
        return (input - double.MinValue) / (double.MaxValue - double.MinValue);
    }

    public double DeNormalize(double input)
    {
        return (input * (double.MaxValue - double.MinValue)) + double.MinValue;
    }

    public void Train(List<TrainingSet> dataSets, int epochs)
    {
        //dataSets = NormalizeTrainingData(dataSets);
        Console.WriteLine("Starting Training");
        for (int i = 0; i < epochs; i++)
        {
            PrintNetwork();
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

        //foreach (var set in dataSets.Take(trainingCount))
        foreach (var set in dataSets)
        {
            Train(set);
        }

        var totalAverageErrors = new List<double>();
        //foreach (var set in dataSets.Skip(trainingCount).Take(validationCount))
        foreach (var set in dataSets)
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
        for (int i = Layers.Count - 2; i > 0; i--)
        {
            //Console.Write($"Training Hidden Layer {i}");
            Layers[i].TrainHiddenNeurons(LearningRate);
            //Console.WriteLine();
        }
        //PrintNetwork();
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

    private void AddLayer(LayerConfiguration configuration)
    {
        var hiddenLayer = new NeuralLayer(configuration.Neurons, configuration.InputFunction, configuration.ActivationFunction, NetworkSettings.Seed);
        if (Layers.Count > 0)
        {
            hiddenLayer.ConnectPreviousLayer(Layers.Last());
        }
        Layers.Add(hiddenLayer);
    }

    public void PrintNetwork()
    {
        var maxNeuronCount = Layers.Select(x => x.Neurons.Count).Max();
        foreach (var layer in Layers)
        {
            var tabAmount = (maxNeuronCount - layer.Neurons.Count) / 2;
            for (int i = 0; i < tabAmount; i++)
            {
                Console.Write("\t");
            }
            layer.PrintLayer();
            Console.WriteLine();
        }
        //Console.ReadKey();
    }
}
