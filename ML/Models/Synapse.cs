public class Synapse
{
    public Neuron InputNeuron { get; private set; }
    public Neuron OutputNeuron { get; private set; }
    public double Weight { get; set; }

    public Synapse(Neuron inputNeuron, Neuron outputNeuron)
    {
        InputNeuron = inputNeuron;
        OutputNeuron = outputNeuron;
        //Weight = Random.Shared.NextDouble() - 0.5;
        Weight = new Random(15).NextDouble() - 0.5;
    }

    public double GetOutput()
    {
        return InputNeuron.CalculateOutput();
    }
}
