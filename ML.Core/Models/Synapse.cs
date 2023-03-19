using ML.Core.Extensions;
using ML.Core.Models;
using System;

public class Synapse : IDisposable
{
    private readonly Random _random;

    public Neuron InputNeuron { get; private set; }
    public Neuron OutputNeuron { get; private set; }
    private double _weight = 0;
    private bool disposedValue = false;

    public double Weight
    {
        get => _weight; set
        {
            PreviousWeight = _weight;
            _weight = value;
        }
    }
    public double PreviousWeight { get; set; }

    public Synapse(Neuron inputNeuron, Neuron outputNeuron, Random random)
    {
        _random = random;
        InputNeuron = inputNeuron;
        OutputNeuron = outputNeuron;
        Weight = random.NextDouble() - 0.5;
    }

    public void Mutate()
    {
        if (_random.CoinFlip(0.2))
        {
            Weight += _random.NextDouble() - 0.5;
        }
    }

    public double GetOutput()
    {
        return InputNeuron.CalculateOutput();
    }

    private void SeverConnection()
    {
        InputNeuron.RemoveSynapse(this);
        OutputNeuron.RemoveSynapse(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!disposedValue)
        {
            SeverConnection();
            disposedValue = true;
        }
    }

    public void Dispose()
    {
        // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
