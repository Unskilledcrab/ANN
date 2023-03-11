public class PowerDifferenceErrorFunction : IErrorFunction
{
    public double CalculateError(double actualOutput, double expectedOutput)
    {
        return Math.Pow(actualOutput - expectedOutput, 2) / 2;
    }

    public double Derivate(double actualOutput, double expectedOutput)
    {
        return actualOutput - expectedOutput;
    }
}