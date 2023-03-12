using System.Collections.Generic;
using System.Linq;
using System;

namespace ML.Core
{
    public static class FakeData
    {
        public static List<TrainingSet> HardCodedSets_3_1()
        {
            return new List<TrainingSet>
        {
            HardCodedSet(Inputs(0,0,0), Outputs(0)),
            HardCodedSet(Inputs(0,0,0), Outputs(0)),
            HardCodedSet(Inputs(0,0,0), Outputs(0)),
            HardCodedSet(Inputs(0,0,0), Outputs(0)),
            HardCodedSet(Inputs(0,0,1), Outputs(1)),
            HardCodedSet(Inputs(0,1,0), Outputs(1)),
            HardCodedSet(Inputs(0,1,1), Outputs(1)),
            HardCodedSet(Inputs(1,0,0), Outputs(1)),
            HardCodedSet(Inputs(1,0,1), Outputs(1)),
            HardCodedSet(Inputs(1,1,1), Outputs(1)),
        };
        }

        public static List<TrainingSet> HardCodedSets_4_3()
        {
            return new List<TrainingSet>
        {
            HardCodedSet(Inputs(0,0,0,1), Outputs(0,0,1)),
            HardCodedSet(Inputs(0,0,1,1), Outputs(-1,0,1)),
            HardCodedSet(Inputs(0,-1,0,1), Outputs(-1,1,1)),
            HardCodedSet(Inputs(0,-1,1,1), Outputs(-1,0,-1)),
            HardCodedSet(Inputs(1,0,0,1), Outputs(-1,0,1)),
            HardCodedSet(Inputs(1,0,1,-1), Outputs(-1,0,1)),
            HardCodedSet(Inputs(1,1,1,1), Outputs(-1,0,1)),
        };
        }

        public static List<double> Inputs(params double[] inputs)
        {
            return inputs.ToList();
        }
        public static List<double> Outputs(params double[] inputs)
        {
            return inputs.ToList();
        }

        public static TrainingSet HardCodedSet(List<double> inputs, List<double> outputs)
        {
            return new TrainingSet
            {
                Inputs = inputs,
                ExpectedOutputs = outputs
            };
        }

        /// <summary>
        /// Use to get a predictable sequence of numbers to train a neural network with
        /// </summary>
        /// <param name="inputs">number of input neurons</param>
        /// <param name="outputs">number of output neurons</param>
        /// <param name="amount">how many training sets to provide</param>
        /// <param name="seed">specify a seed if you want to make a reproducable random set</param>
        /// <returns>predictable sequence of training sets</returns>
        public static List<TrainingSet> GetTrainingData(int inputs, int outputs, int amount = 10000, int? seed = null)
        {
            var random = seed == null ? new Random() : new Random(seed.GetValueOrDefault());
            var fakeData = new List<TrainingSet>();
            for (int i = 0; i < amount; i++)
            {
                fakeData.Add(FakeTrainingSet(random, inputs, outputs));
            }
            return fakeData;
        }

        /// <summary>
        /// This will get the correct number of input and output predictions for a neural network size
        /// </summary>
        /// <param name="random">instance of random if you want to use a specific seed</param>
        /// <param name="inputs">number of inputs to replicate</param>
        /// <param name="outputs">number of outputs to replicate</param>
        /// <returns>a predictable sequence of numbers to train a neural network with</returns>
        public static TrainingSet FakeTrainingSet(Random random, int inputs, int outputs)
        {
            var inputValues = new List<double>();
            var outputValues = new List<double>();

            var rand = random.Next(500) - 250;
            double lowValue = 0;
            for (int i = -25; i < 25; i++)
            {
                var value = i * 10;
                if (rand <= value)
                {
                    lowValue = value;
                    break;
                }
            }
            for (int i = 0; i < inputs; i++)
            {
                inputValues.Add(lowValue);
                //inputValues.Add(7);
            }

            for (int i = 0; i < outputs; i++)
            {
                outputValues.Add(lowValue + 20);
                //outputValues.Add(20);
            }

            return new TrainingSet
            {
                Inputs = inputValues,
                ExpectedOutputs = outputValues
            };
        }
    }
}