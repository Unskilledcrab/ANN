public class NeuralNetwork
{
    public List<NeuralLayer> Layers { get; set; } = new();
    public NeuralLayer InputLayer { get; private set; }
    public NeuralLayer OutputLayer { get; private set; }
    public List<double> Errors { get; private set; } = new();

    public NeuralNetwork(List<LayerConfiguration> configurations)
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
        for (int i = 0; i < epochs; i++)
        {
            foreach (var set in data)
            {
                Train(set);
            }            
        }
    }

    public void Train(TrainingData data)
    {
        SetInputs(data.Inputs);
        BustCache(); // Bust the cache because we are about to calculate the new output
        CalculateError(data.ExpectedOutputs);
        UpdateLayers();
        BustCache(); // Bust the cache again incase we are about to run our first prediction
    }

    private void UpdateLayers()
    {
        throw new NotImplementedException();
    }

    public void CalculateError(List<double> expectedOutputs)
    {
        var actualOutputs = GetOutput();
        Errors = new List<double>();
        for (int i = 0; i < expectedOutputs.Count; i++)
        {
            var error = Math.Pow(actualOutputs[i] - expectedOutputs[i], 2);
            Errors.Add(error);
        }
    }

    /// <summary>
    /// Use to bust the cache so the next time outputs
    /// will be re-calculated
    /// </summary>
    public void BustCache()
    {        
        foreach (var layer in Layers)
        {
            foreach (var neuron in layer.Neurons)
            {
                neuron.IsDirty = true;
            }
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
