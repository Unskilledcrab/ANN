public class WeightedSumFunction : IInputFunction
{
    public double CalculateInput(List<Synapse> inputs)
    {
        return inputs.Select(x => x.Weight * x.GetOutput()).Sum();
    }
}
