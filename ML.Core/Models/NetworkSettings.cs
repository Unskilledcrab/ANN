public class NetworkSettings
{
    public int? Seed { get; set; } = null;
    public double LearningRate { get; set; } = 0.02;
    public IErrorFunction ErrorFunction { get; set; } = new PowerDifferenceErrorFunction();
}
