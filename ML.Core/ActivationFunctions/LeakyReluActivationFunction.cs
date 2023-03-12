public class LeakyReluActivationFunction : IActivationFunction
{
    public double Activate(double input)
    {
        return input >= 0 ? 0.01 * input : input;
    }

    public double Derivate(double input)
    {
        return 0 >= input ? 0.01 : 1;
    }
}