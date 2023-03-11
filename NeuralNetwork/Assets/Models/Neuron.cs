using System;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    private double _cachedValue;

    public IActivationFunction ActivationFunction { get; set; } = new RectifiedActivationFunction();
    public IInputFunction InputFunction { get; set; } = new WeightedSumFunction();
    public List<Synapse> InputSynapses { get; set; } = new();
    public List<Synapse> OutputSynapses { get; set; } = new();
    public double Bias { get; set; }
    public double Value { get; set; }
    public bool IsDirty { get; set; } = true;
    public double Delta { get; private set; }
    public GameObject NeuronObject { get; set; }

    public Neuron(IInputFunction inputFunction, IActivationFunction activationFunction)
    {
        InputFunction = inputFunction;
        ActivationFunction = activationFunction;
        //Bias = Random.Shared.NextDouble() - 0.5;
        Bias = UnityEngine.Random.Range(-0.5f, 0.5f);
    }

    public void ConnectInputNeuron(Neuron neuron)
    {
        var synapse = new Synapse(neuron, this);
        InputSynapses.Add(synapse);
        neuron.OutputSynapses.Add(synapse);
    }

    public void TrainHiddenNeuron(double learningRate)
    {
        //Console.Write(".");
        Delta = 0;
        foreach (var synapse in OutputSynapses)
        {
            Delta += synapse.PreviousWeight * synapse.OutputNeuron.Delta;
        }
        Delta *= ActivationFunction.Derivate(CalculateOutput());
        Bias -= Delta * learningRate;
        foreach (var synapse in InputSynapses)
        {
            synapse.Weight -= Delta * synapse.InputNeuron.CalculateOutput() * learningRate;
        }
    }

    public double TrainOutputNeuron(double expectedOutput, IErrorFunction errorFunction, double learningRate)
    {
        if (OutputSynapses.Count > 0)
        {
            throw new Exception("Attempting to train neuron that is not on the output layer");
        }

        var actualOutput = CalculateOutput();
        var error = errorFunction.CalculateError(actualOutput, expectedOutput);
        Delta = (actualOutput - expectedOutput) * ActivationFunction.Derivate(actualOutput);
        Bias -= Delta * learningRate;
        foreach (var synapse in InputSynapses)
        {
            synapse.Weight -= Delta * synapse.InputNeuron.CalculateOutput() * learningRate;
        }
        // Console.WriteLine($"output: {actualOutput}\t expected: {expectedOutput}\t delta: {Delta}\t bias: {Bias}");
        // Console.ReadKey();
        return error;
    }

    public double CalculateOutput()
    {
        // If we don't have input synapses we are input neurons
        if (InputSynapses.Count == 0)
        {
            return Value;
        }
        if (IsDirty)
        {
            _cachedValue = ActivationFunction.Activate(InputFunction.CalculateInput(InputSynapses) + Bias);
            IsDirty = false;
        }
        return _cachedValue;
    }

    public void PrintNeuron()
    {        
        //Console.Write($"({InputSynapses.Select(s => s.Weight).Sum().ToString("0.0")})\t");
        Console.Write($"({CalculateOutput().ToString("0.000")})\t");
        //Console.Write($"({OutputSynapses.Select(s => s.Weight).Sum().ToString("0.0")})\t");
        //Console.Write($"({Delta.ToString("0.0")})\t");
        //Console.Write($"({Bias.ToString("0.0")})\t");
    }
}