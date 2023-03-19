using System.Collections.Generic;
using System;
using ML.Core.Extensions;

namespace ML.Core.Models
{
    public class Neuron
    {
        private double _cachedValue;

        public IActivationFunction ActivationFunction { get; set; } = new RectifiedActivationFunction();

        private Random _random;

        public IInputFunction InputFunction { get; set; } = new WeightedSumFunction();
        public List<Synapse> InputSynapses { get; set; } = new List<Synapse>();
        public List<Synapse> OutputSynapses { get; set; } = new List<Synapse>();
        public double Bias { get; set; }
        public double Value { get; set; }
        public bool IsDirty { get; set; } = true;
        public double Delta { get; private set; }

        public Neuron(IInputFunction inputFunction, IActivationFunction activationFunction, Random random)
        {
            InputFunction = inputFunction;
            ActivationFunction = activationFunction;
            _random = random;
            Bias = _random.NextDouble() - 0.5;
        }

        public void ConnectInputNeuron(Neuron neuron)
        {
            var synapse = new Synapse(neuron, this, _random);
            InputSynapses.Add(synapse);
            neuron.OutputSynapses.Add(synapse);
        }

        public void TrainHiddenNeuron(double learningRate)
        {
            Delta = 0;
            foreach (var synapse in OutputSynapses)
            {
                Delta += synapse.PreviousWeight * synapse.OutputNeuron.Delta;
            }
            Delta *= ActivationFunction.Derivate(CalculateOutput());
            Bias -= Delta * learningRate;
            foreach (var synapse in InputSynapses)
            {
                synapse.Weight -= Delta * synapse.InputNeuron.CalculateOutput() * learningRate;
            }
        }

        public double TrainOutputNeuron(double expectedOutput, IErrorFunction errorFunction, double learningRate)
        {
            if (OutputSynapses.Count > 0)
            {
                throw new Exception("Attempting to train neuron that is not on the output layer");
            }

            var actualOutput = CalculateOutput();
            var error = errorFunction.CalculateError(actualOutput, expectedOutput);
            Delta = errorFunction.Derivate(actualOutput, expectedOutput) * ActivationFunction.Derivate(actualOutput);
            Bias -= Delta * learningRate;
            foreach (var synapse in InputSynapses)
            {
                synapse.Weight -= Delta * synapse.InputNeuron.CalculateOutput() * learningRate;
            }
            return error;
        }

        public double CalculateOutput()
        {
            // If we don't have input synapses we are input neurons
            if (InputSynapses.Count == 0)
            {
                return Value;
            }
            if (IsDirty)
            {
                _cachedValue = ActivationFunction.Activate(InputFunction.CalculateInput(InputSynapses) + Bias);
                IsDirty = false;
            }
            return _cachedValue;
        }

        public void Mutate()
        {
            if (_random.CoinFlip(0.1))
            {
                foreach (var inputSynapse in InputSynapses)
                {
                    inputSynapse.Mutate();
                }
                foreach (var outputSynapse in OutputSynapses)
                {
                    outputSynapse.Mutate();
                }
                Bias += _random.NextDouble() - 0.5;
            }
        }

        public void SeverInputSynapses()
        {
            for (int i = InputSynapses.Count - 1; i >= 0; i--)
            {
                InputSynapses[i].Dispose();
            }
        }

        public void RemoveSynapse(Synapse synapse)
        {
            if (InputSynapses.Contains(synapse))
                InputSynapses.Remove(synapse);
            if (OutputSynapses.Contains(synapse))
                OutputSynapses.Remove(synapse);
        }
    }
}