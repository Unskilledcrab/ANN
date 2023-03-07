public class PowerDifferenceErrorFunction : IErrorFunction
{
    public double CalculateError(double actualOutput, double expectedOutput)
    {
        return Math.Pow(actualOutput - expectedOutput, 2);
    }
}