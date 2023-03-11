using System;
using UnityEngine;

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

        var renderValue = (float)Math.Abs(value);
        var normalizedRenderValue = Normalize(renderValue);
        //Debug.Log($"Updating the color {normalizedRenderValue}");
        //Debug.Log($"Updating the weight {value}");

        Color color;
        if (normalizedRenderValue > (255/2))
        {
            color = new Color(0, normalizedRenderValue, 0, renderValue - 0.5f);
        }
        else
        {
            color = new Color(normalizedRenderValue, 0, 0, renderValue);
        }
        LineRenderer.material.color = color;
    }

    public float Normalize(float input)
    {
        return input * 255;
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
