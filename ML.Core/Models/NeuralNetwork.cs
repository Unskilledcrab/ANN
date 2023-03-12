using System.Collections.Generic;
using System.Linq;
using System;
using ML.Core.Extensions;

public class NeuralNetwork
{
    public List<NeuralLayer> Layers { get; private set; } = new List<NeuralLayer>();
    public NeuralLayer InputLayer => Layers[0];
    public NeuralLayer OutputLayer => Layers[Layers.Count - 1];
    public List<double> Errors { get; private set; } = new List<double>();

    private Random _random;
    private double inputMin;
    private double inputMax;
    private double outputMin;
    private double outputMax;

    public NetworkSettings NetworkSettings { get; private set; } = new NetworkSettings();
    public List<NetworkStats> NetworkStats { get; private set; } = new List<NetworkStats>();

    private double LearningRate => NetworkSettings.LearningRate;
    private IErrorFunction ErrorFunction => NetworkSettings.ErrorFunction;
    public NeuralNetwork(NetworkConfiguration networkConfiguration)
    {
        var seed = networkConfiguration.NetworkSettings.Seed;
        _random = seed != null ? new Random(seed.Value) : new Random();
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
        SetInputs(inputs.Select(i => i.Normalize(inputMin, inputMax)));
        return GetOutput().Select(o => o.DeNormalize(outputMin, outputMax)).ToList();
    }

    public List<TrainingSet> NormalizeTrainingData(List<TrainingSet> data)
    {
        inputMin = double.MaxValue;
        inputMax = double.MinValue;
        outputMin = double.MaxValue;
        outputMax = double.MinValue;

        foreach (var set in data)
        {
            foreach (var input in set.Inputs)
            {
                if (input < inputMin)
                    inputMin = input;
                if (input > inputMax)
                    inputMax = input;
            }
            foreach (var output in set.ExpectedOutputs)
            {
                if (output < outputMin)
                    outputMin = output;
                if (output > outputMax)
                    outputMax = output;
            }
        }

        for (int i = 0; i < data.Count; i++)
        {
            var set = data[i];
            for (int j = 0; j < set.Inputs.Count; j++)
            {
                var value = set.Inputs[j];
                set.Inputs[j] = value.Normalize(inputMin, inputMax);
            }
            for (int j = 0; j < set.ExpectedOutputs.Count; j++)
            {
                var value = set.ExpectedOutputs[j];
                set.ExpectedOutputs[j] = value.Normalize(outputMin, outputMax);
            }
        }
        return data;
    }

    public double Normalize(double input, double min, double max)
    {
        return (input - min) / (max - min);
    }

    public double DeNormalize(double input, double min, double max)
    {
        return (input * (double.MaxValue - double.MinValue)) + double.MinValue;
    }

    public void Train(List<TrainingSet> dataSets, int epochs)
    {
        dataSets = NormalizeTrainingData(dataSets);
        for (int i = 0; i < epochs; i++)
        {
            PrintNetwork();
            var averageError = SplitAndTrain(dataSets);
            var stats = new NetworkStats
            {
                Epoch = i + 1,
                Error = averageError
            };            
            NetworkStats.Add(stats);
        }
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
        for (int i = Layers.Count - 2; i > 0; i--)
        {
            Layers[i].TrainHiddenNeurons(LearningRate);
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
        var hiddenLayer = new NeuralLayer(configuration.Neurons, configuration.InputFunction, configuration.ActivationFunction, _random);
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
                //Console.Write("\t");
            }
            layer.PrintLayer();
            //Console.WriteLine();
        }
        //Console.ReadKey();
    }
}
