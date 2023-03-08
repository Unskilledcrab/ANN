// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

var inputNeurons = 1;
var outputNeurons = 1;

var builder = new NeuralNetworkBuilder();
builder.LayerConfigurations
    .WithInputs(inputNeurons)
    .WithHiddenLayer(2, activationFunction: new LeakyReluActivationFunction())
    .WithHiddenLayer(15, activationFunction: new LeakyReluActivationFunction())
    .WithHiddenLayer(2, activationFunction: new LeakyReluActivationFunction())
    .WithOutputLayer(outputNeurons, activationFunction: new LeakyReluActivationFunction());

var network = builder.Build();
var fakeData = FakeData.GetTrainingData(inputNeurons, outputNeurons, amount: 1000, seed: 15);

MeasureAccuracy(network, fakeData);
network.Train(fakeData, 5000);
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

static void MeasureAccuracy(NeuralNetwork network, List<TrainingData> fakeData)
{
    var trialData = fakeData.First();
    var predictions = network.Predict(trialData.Inputs);
    NewMethod(trialData, predictions);

}
static void NewMethod(TrainingData trialData, List<double> predictions)
{
    Console.WriteLine($"Inputs: {string.Join(',', trialData.Inputs)}");
    Console.WriteLine($"Expected: {string.Join(',', trialData.ExpectedOutputs)}");
    Console.WriteLine($"Actual: {string.Join(',', predictions)}");
    Console.WriteLine();
}