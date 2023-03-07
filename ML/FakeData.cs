public static class FakeData
{
    public static List<TrainingData> GetTrainingData(int inputs, int outputs, int amount = 10000, int seed = 10)
    {
        var random = new Random(seed);
        var fakeData = new List<TrainingData>();
        for (int i = 0; i < amount; i++)
        {
            fakeData.Add(FakeTrainingData(random, inputs, outputs));
        }
        return fakeData;
    }

    public static TrainingData FakeTrainingData(Random random, int inputs, int outputs)
    {
        var inputValues = new List<double>();
        var outputValues = new List<double>();

        for (int i = 0; i < inputs; i++)
        {
            var lowValue = Math.Min(3, random.NextDouble());
            inputValues.Add(lowValue);
        }

        for (int i = 0; i < outputs; i++)
        {
            var highValue = Math.Max(7, random.NextDouble());
            outputValues.Add(highValue);
        }

        return new TrainingData
        {
            Inputs = inputValues,
            ExpectedOutputs = outputValues
        };
    }
}