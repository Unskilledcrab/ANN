// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

var inputNeurons = 3;
var outputNeurons = 1;

var network = NeuralNetworkBuilder
    .CreateNetwork()
    .WithSettings(0.5, new PowerDifferenceErrorFunction())
    .WithInputLayer(inputNeurons)
    .WithHiddenLayer(new LayerConfiguration { Neurons = 2 })
    //.WithHiddenLayer(new LayerConfiguration { Neurons = 15 })
    // .WithHiddenLayer(new LayerConfiguration { Neurons = 8 })
    .WithOutputLayer(new LayerConfiguration { Neurons = outputNeurons, ActivationFunction = new SigmoidActivationFunction() });

//var fakeData = FakeData.GetTrainingData(inputNeurons, outputNeurons, amount: 2000, seed: 15);
var fakeData = FakeData.HardCodedSets();

MeasureAccuracy(network, fakeData);
network.Train(fakeData, 1000);
MeasureAccuracy(network, fakeData);

var inputs = new List<double>();
while (true)
{
    Console.WriteLine("Input values:");
    inputs.Clear();
    for (int i = 0; i < inputNeurons; i++)
    {
        Console.Write($"Neuron {i+1}: ");
        var input = Console.ReadLine();
        if (string.IsNullOrEmpty(input))
        {
            break;
        }
        if (double.TryParse(input, out double result))
        {
            inputs.Add(result);
        }
        else
        {
            Console.WriteLine($"'{input}' is not a valid double");
            i--;
        }
    }
    if (inputs.Count < inputNeurons) break;
    var predictions = network.Predict(inputs);
    Console.WriteLine();
    Console.WriteLine($"Inputs: {string.Join(',', inputs)}");
    Console.WriteLine($"Predictions: {string.Join(',', predictions)}");
    Console.WriteLine();
}

static void MeasureAccuracy(NeuralNetwork network, List<TrainingSet> fakeData)
{
    var trialData = fakeData.First();
    var predictions = network.Predict(trialData.Inputs);
    NewMethod(trialData, predictions);

}
static void NewMethod(TrainingSet trialData, List<double> predictions)
{
    Console.WriteLine($"Inputs: {string.Join(',', trialData.Inputs)}");
    Console.WriteLine($"Expected: {string.Join(',', trialData.ExpectedOutputs)}");
    Console.WriteLine($"Actual: {string.Join(',', predictions)}");
    Console.WriteLine();
}