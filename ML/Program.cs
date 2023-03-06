// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

var inputNeurons = 4;
var outputNeurons = 5;

var builder = new NeuralNetworkBuilder();
builder.LayerConfigurations
    .WithInputs(inputNeurons)
    .WithHiddenLayer(5)
    .WithHiddenLayer(3)
    .WithOutputLayer(outputNeurons);

var network = builder.Build();

var fakeData = FakeData.GetTrainingData(inputNeurons, outputNeurons);

network.Train(fakeData, 1000);