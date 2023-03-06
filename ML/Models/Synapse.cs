public class Synapse
{
    public Neuron InputNeuron { get; private set; }
    public Neuron OutputNeuron { get; private set; }
    public double Weight { get; set; }

    public Synapse(Neuron inputNeuron, Neuron outputNeuron)
    {
        InputNeuron = inputNeuron;
        OutputNeuron = outputNeuron;
    }

    public double GetOutput()
    {
        return InputNeuron.CalculateOutput();
    }

    public void UpdateWeight(double learningRate, double delta)
    {
        Weight += learningRate * delta;
    }
}
