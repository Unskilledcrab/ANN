using System;
using UnityEngine;
using UnityEngine.Windows;

public class Synapse
{
    public Neuron InputNeuron { get; private set; }
    public Neuron OutputNeuron { get; private set; }
    private double _weight = 0;
    public LineRenderer LineRenderer { get; set; }

    public double Weight
    {
        get => _weight; set
        {
            PreviousWeight = _weight;
            _weight = value;
            UpdateColor(value);
        }
    }

    private void UpdateColor(double value)
    {
        if (LineRenderer == null) return;

        var alpha = Normalize(Math.Abs(value), 0, 5);
        var brightness = Brightness(alpha);

        Color color;
        if (value > 0)
        {
            color = new Color(0, brightness, 0, alpha);
        }
        else
        {
            color = new Color(brightness, 0, 0, alpha);
        }
        LineRenderer.material.color = color;
    }

    public float Normalize(double value, double min, double max)
    {
        return (float)((value - min) / (max - min));
    }

    public float Brightness(float alpha)
    {
        return alpha * (255 / 5);
    }

    public double PreviousWeight { get; set; }

    public Synapse(Neuron inputNeuron, Neuron outputNeuron)
    {
        InputNeuron = inputNeuron;
        OutputNeuron = outputNeuron;
        Weight = UnityEngine.Random.Range(-0.5f, 0.5f);
        //Weight = new System.Random(15).NextDouble() - 0.5;
    }

    public double GetOutput()
    {
        return InputNeuron.CalculateOutput();
    }
}
