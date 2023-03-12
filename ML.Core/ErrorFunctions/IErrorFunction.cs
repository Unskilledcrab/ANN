public interface IErrorFunction
{
    double CalculateError(double actualOutput, double expectedOutput);
    double Derivate(double actualOutput, double expectedOutput);
}