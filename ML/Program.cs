// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

var inputNeurons = 3;
var outputNeurons = 3;

var network = NeuralNetworkBuilder
    .CreateNetwork()
    .WithSettings(0.02, new PowerDifferenceErrorFunction())
    .WithInputLayer(inputNeurons)
    .WithHiddenLayer(new LayerConfiguration { Neurons = 8 })
    .WithHiddenLayer(new LayerConfiguration { Neurons = 15 })
    .WithHiddenLayer(new LayerConfiguration { Neurons = 8 })
    .WithOutputLayer(new LayerConfiguration { Neurons = outputNeurons, ActivationFunction = new LeakyReluActivationFunction() });

var fakeData = FakeData.GetTrainingData(inputNeurons, outputNeurons, amount: 2000, seed: 15);

MeasureAccuracy(network, fakeData);
network.Train(fakeData, 1000);
MeasureAccuracy(network, fakeData);

while (true)
{
    Console.Write("Input a value:");
    var input = Console.ReadLine();
    if (string.IsNullOrEmpty(input))
    {
        break;
    }

    if (double.TryParse(input, out double result))
    {
        var inputs = new List<double>();
        for (int i = 0; i < inputNeurons; i++)
        {
            inputs.Add(result);
        }
        var predictions = network.Predict(inputs);
        Console.WriteLine($"Predictions: {string.Join(',', predictions)}");
    }
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