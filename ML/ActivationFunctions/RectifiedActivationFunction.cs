public class RectifiedActivationFunction : IActivationFunction
{
    public double Activate(double input)
    {
        return Math.Max(0, input);
    }

    public double Derivate(double input)
    {
        return input > 0 ? 1 : 0;
    }
}