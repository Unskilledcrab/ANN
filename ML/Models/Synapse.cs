public class Synapse
{
    public Neuron InputNeuron { get; private set; }
    public Neuron OutputNeuron { get; private set; }
    private double _weight = 0;
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
        InputNeuron = inputNeuron;
        OutputNeuron = outputNeuron;
        Weight = random.NextDouble() - 0.5;
    }

    public double GetOutput()
    {
        return InputNeuron.CalculateOutput();
    }
}
