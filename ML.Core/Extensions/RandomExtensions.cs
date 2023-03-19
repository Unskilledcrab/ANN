using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Core.Extensions
{
    public static class RandomExtensions
    {
        public static bool CoinFlip(this Random random, double percentChance)
        {
            return random.NextDouble() < percentChance;
        }
    }
}
