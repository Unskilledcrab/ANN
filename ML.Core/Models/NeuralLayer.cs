using System.Collections.Generic;
using System;
using ML.Core.Models;
using ML.Core.Extensions;

public class NeuralLayer
{
    private readonly Random _random;

    public List<Neuron> Neurons { get; set; } = new List<Neuron>();

    public NeuralLayer(LayerConfiguration layerConfiguration, Random random)
    {
        _random = random;
        for (int i = 0; i < layerConfiguration.Neurons; i++)
        {
            Neurons.Add(new Neuron(layerConfiguration.InputFunction, layerConfiguration.ActivationFunction, random));
        }
    }

    public void ConnectPreviousLayer(NeuralLayer layer)
    {
        foreach (var neuron in Neurons)
        {
            foreach (var InputNeuron in layer.Neurons)
            {
                neuron.ConnectInputNeuron(InputNeuron);
            }
        }
    }

    public void TrainHiddenNeurons(double learningRate)
    {
        foreach (var neuron in Neurons)
        {
            neuron.TrainHiddenNeuron(learningRate);
        }
    }

    public double TrainOutputNeurons(List<double> expectedOutputs, IErrorFunction errorFunction, double learningRate)
    {
        double error = 0;
        for (int i = 0; i < expectedOutputs.Count; i++)
        {            
            error += Neurons[i].TrainOutputNeuron(expectedOutputs[i], errorFunction, learningRate);
        }
        return error;
    }

    public void Mutate()
    {
        if (_random.CoinFlip(0.05))
        {
            foreach (var neuron in Neurons)
            {
                neuron.Mutate();
            }
        }
    }

    public void SeverInputLayerConnection()
    {
        foreach (var neuron in Neurons)
        {
            neuron.SeverInputSynapses();
        }
    }

    public void SetDirty(bool isDirty)
    {
        foreach (var neuron in Neurons)
        {
            neuron.IsDirty = isDirty;
        }
    }
}