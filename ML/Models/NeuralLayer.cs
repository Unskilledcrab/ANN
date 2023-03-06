public class NeuralLayer
{
    public List<Neuron> Neurons { get; set; } = new();

    public NeuralLayer(int neurons, IInputFunction inputFunction, IActivationFunction activationFunction)
    {
        for (int i = 0; i < neurons; i++)
        {
            Neurons.Add(new Neuron(inputFunction, activationFunction));
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
}