public class SigmoidActivationFunction : IActivationFunction
{
    public double Activate(double input)
    {
        var exp = Math.Exp(input);
        return exp / (1 + exp);
    }

    public double Derivate(double input)
    {
        return input * (1 - input);
    }
}