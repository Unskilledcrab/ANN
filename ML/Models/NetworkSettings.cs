public class NetworkSettings
{
    public Random Seed { get; set; } = Random.Shared;
    public double LearningRate { get; set; } = 0.02;
    public IErrorFunction ErrorFunction { get; set; } = new PowerDifferenceErrorFunction();
}
