public class Neuron
{
    public IActivationFunction ActivationFunction { get; set; } = new RectifiedActivationFunction();
    public IInputFunction InputFunction { get; set; } = new WeightedSumFunction();
    public List<Synapse> InputSynapses { get; set; } = new();
    public List<Synapse> OutputSynapses { get; set; } = new();
    public double Bias { get; set; }
    public double Value { get; set; }

    public Neuron(IInputFunction inputFunction, IActivationFunction activationFunction)
    {
        InputFunction = inputFunction;
        ActivationFunction = activationFunction;
    }

    public void ConnectInputNeuron(Neuron neuron)
    {
        var synapse = new Synapse(neuron, this);
        InputSynapses.Add(synapse);
        neuron.OutputSynapses.Add(synapse);
    }

    public double TrainNeuron(double expectedOutput, IErrorFunction errorFunction)
    {
        var actualOutput = CalculateOutput();
        var cost = errorFunction.CalculateError(actualOutput, expectedOutput);

        var change = (actualOutput - expectedOutput) * ActivationFunction.Derivate(actualOutput);
        Bias += change * 0.02;
        foreach (var synapse in InputSynapses)
        {
            
        }
        return cost;
    }

    public double CalculateOutput()
    {
        // If we don't have input synapses we are input neurons
        if (InputSynapses.Count == 0)
        {
            return Value;
        }

        return ActivationFunction.Activate(InputFunction.CalculateInput(InputSynapses)) + Bias;
    }
}
