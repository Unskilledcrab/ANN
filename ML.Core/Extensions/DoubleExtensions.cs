namespace ML.Core.Extensions
{
    public static class DoubleExtensions
    {
        public static double Normalize(this double input, double min, double max)
        {
            if (min == max) return 1;
            return (input - min) / (max - min);
        }

        public static double DeNormalize(this double input, double min, double max)
        {
            if (min == max) return 1;
            return (input * (max - min)) + min;
        }
    }
}
