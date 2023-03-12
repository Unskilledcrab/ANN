using System.Collections.Generic;
using System;
using ML.Core.Models;

public class NeuralLayer
{
    public List<Neuron> Neurons { get; set; } = new List<Neuron>();

    public NeuralLayer(int neurons, IInputFunction inputFunction, IActivationFunction activationFunction, Random random)
    {
        for (int i = 0; i < neurons; i++)
        {
            Neurons.Add(new Neuron(inputFunction, activationFunction, random));
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

    public void SetDirty(bool isDirty)
    {
        foreach (var neuron in Neurons)
        {
            neuron.IsDirty = isDirty;
        }
    }

    public void PrintLayer()
    {
        foreach (var neuron in Neurons)
        {
            neuron.PrintNeuron();
        }
    }
}