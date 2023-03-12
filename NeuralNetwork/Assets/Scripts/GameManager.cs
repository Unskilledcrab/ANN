using ML.Core;
using ML.Core.Extensions;
using ML.Core.Models;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UIElements;

public class GameManager : MonoBehaviour
{
    [SerializeField] private NeuronController neuronPrefab;
    [SerializeField] private LineRenderer lineRendererPrefab;
    [SerializeField] private double learningRate = 0.5;
    [SerializeField] private List<int> hiddenLayerNeurons = new List<int>() { 4, 5 };
    [SerializeField] private Material transparentMaterial;
    [SerializeField] private float offset = 2.5f;
    [SerializeField] private float trainingInterval = 0.005f;
    [SerializeField] private TextMeshProUGUI topStatusBar;
    [SerializeField] private TextMeshProUGUI bottomStatusBar;
    [SerializeField] private VerticalLayoutGroup neuronPanel;
    [SerializeField] private VerticalLayoutGroup inputSynapsePanel;
    [SerializeField] private VerticalLayoutGroup outputSynapsePanel;

    private NeuralNetwork network;
    private int maxNeuronsInLayer;
    private List<List<NeuronController>> networkNeurons;
    private int trainingCount;
    private Dictionary<Synapse, LineRenderer> synapseDictionary = new();
    private Dictionary<Neuron, NeuronController> neuronDictionary = new();

    public int InputNeurons { get; set; } = 3;
    public int OutputNeurons { get; set; } = 1;


    // Start is called before the first frame update
    void Start()
    {
        var builder = NeuralNetworkBuilder
            .CreateNetwork()
            .WithSettings(s => s.LearningRate = learningRate)
            .WithInputLayer(InputNeurons);

        foreach (var hiddenLayerNeuron in hiddenLayerNeurons)
        {
            if (hiddenLayerNeuron > 0)
                builder.WithHiddenLayer(l => l.Neurons = hiddenLayerNeuron);
        }

        network = builder
            .WithOutputLayer(l => { l.Neurons = OutputNeurons; l.ActivationFunction = new SigmoidActivationFunction(); })
            .Build();

        maxNeuronsInLayer = network.Layers.Select(l => l.Neurons.Count).Max();
        networkNeurons = DrawNeuralNetwork(network);
        DrawConnections();
        trainingCount = 0;
        InvokeRepeating("TrainNetwork", trainingInterval, trainingInterval);
    }

    private void TrainNetwork()
    {
        var trainingSets = FakeData.HardCodedSets_3_1();
        GetPredictionStatus(trainingSets);
        network.Train(trainingSets, 1);
        bottomStatusBar.text = $"Training Count: {trainingCount}";
        trainingCount++;
    }

    private void GetPredictionStatus(List<TrainingSet> trainingSets)
    {
        var firstTrainingSet = trainingSets.First();
        var lastTrainingSet = trainingSets.Last();
        var predictionFirst = network.Predict(firstTrainingSet.Inputs);
        var predictionSecond = network.Predict(lastTrainingSet.Inputs);
        var firstPrediction = $"Inputs: {string.Join(',', firstTrainingSet.Inputs)}\tExpected: {string.Join(',', firstTrainingSet.ExpectedOutputs)}\tPredicted: {string.Join(',', predictionFirst.Select(p => p.ToString("0.0000")))}";
        var secondPrediction = $"Inputs: {string.Join(',', lastTrainingSet.Inputs)}\tExpected: {string.Join(',', lastTrainingSet.ExpectedOutputs)}\tPredicted: {string.Join(',', predictionSecond.Select(p => p.ToString("0.0000")))}";
        topStatusBar.text = $"{firstPrediction}\n{secondPrediction}";
    }

    private void DrawConnections()
    {
        for (int currentLayerIndex = 0; currentLayerIndex < networkNeurons.Count - 1; currentLayerIndex++)
        {
            var nextLayerIndex = currentLayerIndex + 1;
            for (int currentNeuronIndex = 0; currentNeuronIndex < networkNeurons[currentLayerIndex].Count; currentNeuronIndex++)
            {
                for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < networkNeurons[nextLayerIndex].Count; nextLayerNeuronIndex++)
                {
                    var lineRenderer = CreateLineRenderer();

                    var synapse = network.Layers[currentLayerIndex].Neurons[currentNeuronIndex].OutputSynapses[nextLayerNeuronIndex];
                    synapseDictionary.Add(synapse, lineRenderer);
                    var currentNeuron = network.Layers[currentLayerIndex].Neurons[currentNeuronIndex];
                    var nextLayerNeuron = network.Layers[nextLayerIndex].Neurons[nextLayerNeuronIndex];

                    var currenNeuronGO = neuronDictionary[currentNeuron];
                    var nextLayerNeuronGO = neuronDictionary[nextLayerNeuron];

                    lineRenderer.SetPosition(0, currenNeuronGO.transform.position);
                    lineRenderer.SetPosition(1, nextLayerNeuronGO.transform.position);
                }
            }
        }
    }

    private LineRenderer CreateLineRenderer()
    {
        return Instantiate(lineRendererPrefab);
    }

    private List<List<NeuronController>> DrawNeuralNetwork(NeuralNetwork network)
    {
        var networkNeurons = new List<List<NeuronController>>();
        for (int layerIndex = 0; layerIndex < network.Layers.Count; layerIndex++)
        {
            var neurons = DrawLayer(layerIndex, network.Layers[layerIndex]);
            networkNeurons.Add(neurons);
        }
        return networkNeurons;
    }

    private List<NeuronController> DrawLayer(int layerIndex, NeuralLayer neuralLayer)
    {
        var neurons = new List<NeuronController>();
        for (int neuronIndex = 0; neuronIndex < neuralLayer.Neurons.Count; neuronIndex++)
        {
            var neuron = DrawNeuron(layerIndex, neuronIndex);
            neuronDictionary.Add(neuralLayer.Neurons[neuronIndex], neuron);
            neurons.Add(neuron);
        }
        var totalOffset = maxNeuronsInLayer - neurons.Count;
        var actualOffset = totalOffset / 2.0f;
        foreach (var neuron in neurons)
        {
            neuron.transform.Translate(actualOffset * offset * Vector3.up);
        }
        return neurons;
    }

    private NeuronController DrawNeuron(int layerIndex, int neuronIndex)
    {
        var unityNeuron = CreateNeuron();
        unityNeuron.transform.position = new Vector3(layerIndex * offset, neuronIndex * offset, 0);
        return unityNeuron;
    }

    NeuronController CreateNeuron()
    {
        var newNeuron = Instantiate(neuronPrefab);
        newNeuron.MouseEnter += OnNeuronEnter;
        return newNeuron;
    }

    private void OnNeuronEnter(object sender, EventArgs args)
    {
        var neuronController = (NeuronController)sender;
        var neuron = neuronDictionary.Where(n => n.Value == neuronController).FirstOrDefault().Key;

        if (neuron != null)
            UpdateStatPanels(neuron);
    }

    private void UpdateStatPanels(Neuron selectedNeuron)
    {
        UpdateNeuronPanel(selectedNeuron);
        UpdateSynapsePanel(selectedNeuron, inputSynapsePanel, true);
        UpdateSynapsePanel(selectedNeuron, outputSynapsePanel, false);
    }

    private void UpdateNeuronPanel(Neuron selectedNeuron)
    {
        var bias = selectedNeuron.Bias;
        var delta = selectedNeuron.Delta;
        var output = selectedNeuron.CalculateOutput();
        DeleteAllChildComponents(neuronPanel.gameObject);

        var HeaderObject = Instantiate(new GameObject(), neuronPanel.transform);
        var headerText = HeaderObject.AddComponent<TextMeshProUGUI>();
        headerText.fontSize = 16;
        headerText.fontStyle = FontStyles.Bold;
        headerText.text = $"Neuron Stats";

        var neuronObject = Instantiate(new GameObject(), neuronPanel.transform);
        var neuronText = neuronObject.AddComponent<TextMeshProUGUI>();
        neuronText.fontSize = 14;
        neuronText.text = $"Bias: {bias:0.00000}\nDelta: {delta:0.00000}\nOutput: {output:0.00000}";
    }

    private void UpdateSynapsePanel(Neuron selectedNeuron, VerticalLayoutGroup synapsePanel, bool isInput)
    {
        DeleteAllChildComponents(synapsePanel.gameObject);

        var HeaderObject = Instantiate(new GameObject(), synapsePanel.transform);
        var headerText = HeaderObject.AddComponent<TextMeshProUGUI>();
        headerText.fontSize = 16;
        headerText.fontStyle = FontStyles.Bold;
        headerText.text = $"Weights";

        List<Synapse> synapses;
        if (isInput)
        {
            synapses = selectedNeuron.InputSynapses;
            headerText.text = $"Input {headerText.text}";
        }            
        else
        {
            synapses = selectedNeuron.OutputSynapses;
            headerText.text = $"Output {headerText.text}";
        }

        foreach (var synapse in synapses)
        {
            var synapseObject = Instantiate(new GameObject(), synapsePanel.transform);
            var tmpText = synapseObject.AddComponent<TextMeshProUGUI>();
            tmpText.fontSize = 14;
            tmpText.text = $"{synapse.Weight:0.00000}";
        }
    }

    private void DeleteAllChildComponents(GameObject gameObject)
    {
        foreach (var component in gameObject.GetComponentsInChildren<Transform>().Skip(1))
        {
            Destroy(component.gameObject);
        }
    }

    // Update is called once per frame
    void Update()
    {
        UpdateConnections();
    }

    private void UpdateConnections()
    {
        for (int currentLayerIndex = 0; currentLayerIndex < networkNeurons.Count - 1; currentLayerIndex++)
        {
            for (int currentNeuronIndex = 0; currentNeuronIndex < networkNeurons[currentLayerIndex].Count; currentNeuronIndex++)
            {
                var currentNeuron = network.Layers[currentLayerIndex].Neurons[currentNeuronIndex];
                foreach (var outputSynapse in currentNeuron.OutputSynapses)
                {
                    var lineRenderer = synapseDictionary[outputSynapse];

                    double max = 2;
                    var alpha = (float)Math.Abs(outputSynapse.Weight).Normalize(0, max);
                    var brightness = Brightness(alpha, (float)max);

                    Color color;
                    if (outputSynapse.Weight > 0)
                    {
                        color = new Color(0, brightness, 0, alpha);
                    }
                    else
                    {
                        color = new Color(brightness, 0, 0, alpha);
                    }
                    lineRenderer.material.color = color;
                }
            }
        }
    }

    private float Brightness(float alpha, float max)
    {
        return alpha * (255 / max);
    }
}
