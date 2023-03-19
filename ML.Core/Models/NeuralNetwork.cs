using System.Collections.Generic;
using System.Linq;
using System;
using ML.Core.Extensions;
using Newtonsoft.Json.Linq;
using ML.Core.Models;

public class NeuralNetwork
{
    public List<NeuralLayer> Layers { get; private set; } = new List<NeuralLayer>();
    public NeuralLayer InputLayer => Layers[0];
    public NeuralLayer OutputLayer => Layers[Layers.Count - 1];
    public List<double> Errors { get; private set; } = new List<double>();
    public NetworkSettings NetworkSettings { get; private set; } = new NetworkSettings();
    public List<NetworkStats> NetworkStats { get; private set; } = new List<NetworkStats>();

    private Random _random;
    private double inputMin;
    private double inputMax;
    private double outputMin;
    private double outputMax;

    private double LearningRate => NetworkSettings.LearningRate;
    private IErrorFunction ErrorFunction => NetworkSettings.ErrorFunction;
    public NeuralNetwork(NetworkConfiguration networkConfiguration)
    {
        var seed = networkConfiguration.NetworkSettings.Seed;
        _random = seed != null ? new Random(seed.Value) : new Random();
        NetworkSettings = networkConfiguration.NetworkSettings;
        SetupLayers(networkConfiguration.LayerConfigurations);
    }

    public NeuralNetwork(string jsonSerializedNetwork)
    {
        _random = new Random();
        LoadSerializedNetwork(jsonSerializedNetwork);
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

    private List<TrainingSet> NormalizeTrainingData(List<TrainingSet> data)
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

    public void Train(List<TrainingSet> dataSets, int epochs)
    {
        dataSets = NormalizeTrainingData(dataSets);
        for (int i = 0; i < epochs; i++)
        {
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

    private void Train(TrainingSet set)
    {
        SetInputs(set.Inputs);
        UpdateLayers(set.ExpectedOutputs);
    }

    private void UpdateLayers(List<double> expectedOutputs)
    {
        UpdateOutputLayer(expectedOutputs);
        UpdateHiddenLayers();
    }

    private void UpdateHiddenLayers()
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
    }

    private void UpdateOutputLayer(List<double> expectedOutputs)
    {
        OutputLayer.TrainOutputNeurons(expectedOutputs, ErrorFunction, LearningRate);
    }

    private void CalculateError(List<double> expectedOutputs)
    {
        Errors.Clear();
        var actualOutputs = GetOutput();
        for (int i = 0; i < expectedOutputs.Count; i++)
        {            
            var error = ErrorFunction.CalculateError(actualOutputs[i], expectedOutputs[i]);
            Errors.Add(error);
        }
    }

    private void SetInputs(IEnumerable<double> inputs)
    {
        var index = 0;
        foreach (var input in inputs)
        {
            InputLayer.Neurons[index].Value = input;
            index++;
        }
        SetDirty(true);
    }

    private List<double> GetOutput()
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

    private void AddLayer(LayerConfiguration configuration, int? index = null)
    {
        var hiddenLayer = new NeuralLayer(configuration, _random);

        if (Layers.Count > 0)
        {
            var previousLayer = Layers.Last();
            if (index != null)
            {
                previousLayer = Layers[index.Value - 1];
                Layers[index.Value].SeverInputLayerConnection();
            }
            hiddenLayer.ConnectPreviousLayer(previousLayer);
        }

        if (index == null)
        {
            Layers.Add(hiddenLayer);
        }
        else
        {
            Layers.Insert(index.Value, hiddenLayer);
            if (index.Value != Layers.Count - 1)
            {
                Layers[index.Value + 1].ConnectPreviousLayer(hiddenLayer);
            }
        }
    }

    public void Mutate()
    {
        MutateNetworkStructure();
        MutateLayers();
    }

    private void MutateNetworkStructure()
    {
        if (_random.CoinFlip(0.02))
        {
            var inputNeuronCount = InputLayer.Neurons.Count;
            var neuronCount = _random.Next(1, inputNeuronCount + 1);
            var newLayerConfig = new LayerConfiguration { Neurons = neuronCount };

            // Insert somewhere between the first and last layer
            var insertIndex = _random.Next(1, Layers.Count - 1);
            AddLayer(newLayerConfig, insertIndex);
        }
    }

    private void MutateLayers()
    {
        foreach (var layer in Layers)
        {
            layer.Mutate();
        }
    }

    public void LoadSerializedNetwork(string json)
    {
        Layers.Clear();
        var obj = JObject.Parse(json);
        var layerObjs = (JArray?)obj[nameof(Layers)];

        if (layerObjs is null)
            return;

        int layerIndex = 0;
        foreach (var layerObj in layerObjs)
        {
            var neuronObjs = (JArray?)layerObj[nameof(NeuralLayer.Neurons)];
            if (neuronObjs is null)
                return;

            var neuronCount = neuronObjs.Count();
            AddLayer(new LayerConfiguration { Neurons = neuronCount });

            int neuronIndex = 0;
            var layer = Layers[layerIndex];
            foreach (var neuronObj in neuronObjs)
            {
                var neuron = layer.Neurons[neuronIndex];
                var bias = (double?)neuronObj[nameof(Neuron.Bias)];
                var inputSynapses = (JArray?)neuronObj[nameof(Neuron.InputSynapses)];

                neuron.Bias = bias.GetValueOrDefault();
                if (inputSynapses is null)
                    return;

                int synapseIndex = 0;
                foreach (var synapseObj in inputSynapses)
                {
                    var synapse = neuron.InputSynapses[synapseIndex];
                    var weight = (double?)synapseObj[nameof(Synapse.Weight)];
                    synapse.Weight = weight.GetValueOrDefault();
                    synapseIndex++;
                }
                neuronIndex++;
            }
            layerIndex++;
        }
    }

    public string SerializeNetwork()
    {
        var networkObj = new JObject();
        var layerObjs = new JArray();
        foreach (var layer in Layers)
        {
            var layerObj = new JObject();
            var neuronObjs = new JArray();
            foreach (var neuron in layer.Neurons)
            {
                var neuronObj = new JObject();
                var inputSynapseObjs = new JArray();
                foreach (var inputSynapse in neuron.InputSynapses)
                {
                    var synapseObj = new JObject();
                    synapseObj[nameof(Synapse.Weight)] = inputSynapse.Weight;
                    inputSynapseObjs.Add(synapseObj);
                }
                neuronObj[nameof(Neuron.InputSynapses)] = inputSynapseObjs;
                neuronObj[nameof(Neuron.Bias)] = neuron.Bias;
                neuronObjs.Add(neuronObj);
            }
            layerObj[nameof(NeuralLayer.Neurons)] = neuronObjs;
            layerObjs.Add(layerObj);
        }
        networkObj[nameof(Layers)] = layerObjs;

        var json = networkObj.ToString();
        return json;
    }
}
