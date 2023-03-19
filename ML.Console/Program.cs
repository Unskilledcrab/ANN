using ML.Core;

var inputNeurons = 3;
var outputNeurons = 1;

var network = NeuralNetworkBuilder
    .CreateNetwork()
    .WithSettings(s => { s.LearningRate = 0.5; s.Seed = 1; })
    .WithInputLayer(inputNeurons)
    .WithHiddenLayer(l => l.Neurons = 3)
    .WithHiddenLayer(l => l.Neurons = 3)
    //.WithHiddenLayer(l => l.Neurons = 15)    
    .WithOutputLayer(l => {l.Neurons = outputNeurons; l.ActivationFunction = new SigmoidActivationFunction(); })
    .Build();

//var fakeData = FakeData.GetTrainingData(inputNeurons, outputNeurons, amount: 2000, seed: 15);
var fakeData = FakeData.HardCodedSets_3_1();

MeasureAccuracy(network, fakeData);
for (int i = 0; i < 400; i++)
{
    //var loadedJson = File.ReadAllText($"C:\\Users\\jb090\\Desktop\\Testing\\test_again{i}.json");
    //network.LoadSerializedNetwork(loadedJson);
    network.Mutate();
    var json = network.SerializeNetwork();
    File.WriteAllText($"C:\\Users\\jb090\\Desktop\\Testing\\test{i}.json", json);
}
//network.Train(fakeData, 1000);
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
    OutputNetworkStructure(network);
    OutputResults(trialData, predictions);
}

static void OutputNetworkStructure(NeuralNetwork network)
{
    Console.WriteLine($"Layer Count: {network.Layers.Count}");
    Console.WriteLine($"{string.Join(Environment.NewLine, 
            network.Layers.Select(l => 
                $"[{string.Join(',', l.Neurons.Select(n => 
                    $"(IS: {n.InputSynapses.Count}, OS: {n.OutputSynapses.Count})"))}]"
            )
        )}"
    );
}

static void OutputResults(TrainingSet trialData, List<double> predictions)
{
    Console.WriteLine($"Inputs: {string.Join(',', trialData.Inputs)}");
    Console.WriteLine($"Expected: {string.Join(',', trialData.ExpectedOutputs)}");
    Console.WriteLine($"Actual: {string.Join(',', predictions)}");
    Console.WriteLine();
}