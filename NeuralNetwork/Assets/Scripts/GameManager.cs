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
    [SerializeField] private TextMeshProUGUI bottomStatusBar;
    [SerializeField] private VerticalLayoutGroup neuronPanel;
    [SerializeField] private VerticalLayoutGroup inputSynapsePanel;
    [SerializeField] private VerticalLayoutGroup outputSynapsePanel;
    [SerializeField] private VerticalLayoutGroup layerPanel;
    [SerializeField] private VerticalLayoutGroup trainingPanel;
    [SerializeField] private TextMeshProUGUI trainingText;
    [SerializeField] private HorizontalLayoutGroup networkPanel;
    [SerializeField] private Canvas canvas;

    private NeuralNetwork network;
    private int maxNeuronsInLayer;
    private List<List<NeuronController>> networkNeurons;
    private int trainingCount;
    private Dictionary<Synapse, LineRenderer> synapseDictionary = new();
    private Dictionary<Neuron, NeuronController> neuronDictionary = new();
    private bool isTraining = true;

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

        DeleteAllChildComponents(networkPanel.gameObject);
        maxNeuronsInLayer = network.Layers.Select(l => l.Neurons.Count).Max();
        networkNeurons = DrawNeuralNetwork(network);
        DrawConnections();
        trainingCount = 0;
        InvokeRepeating(nameof(TrainNetwork), trainingInterval, trainingInterval);
    }

    public void ToggleTraining()
    {
        if (isTraining)
        {
            CancelInvoke(nameof(TrainNetwork));
            isTraining = false;
        }
        else
        {
            InvokeRepeating(nameof(TrainNetwork), trainingInterval, trainingInterval);
            isTraining = true;
        }
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
        DeleteAllChildComponents(trainingPanel.gameObject);
        foreach (var item in trainingSets)
        {
            var prediction = network.Predict(item.Inputs);
            var predictionStat = $"Inputs: {string.Join(',', item.Inputs)} Expected: {string.Join(',', item.ExpectedOutputs)} Predicted: {string.Join(',', prediction.Select(p => p.ToString("0.0000")))}";

            var newTrainingStat = Instantiate(trainingText, trainingPanel.transform);
            newTrainingStat.enabled = true;
            newTrainingStat.text = predictionStat;
        }
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

                    var currentNeuronGO = neuronDictionary[currentNeuron];
                    var nextLayerNeuronGO = neuronDictionary[nextLayerNeuron];

                    //var currentPosition = Camera.main.WorldToScreenPoint(currentNeuronGO.localPosition);
                    //var nextPosition = Camera.main.WorldToScreenPoint(nextLayerNeuronGO.localPosition);
                    lineRenderer.SetPosition(0, currentNeuronGO.transform.position);
                    lineRenderer.SetPosition(1, nextLayerNeuronGO.transform.position);
                }
            }
        }
    }

    private Vector3 GetCanvasPosition(Vector3 position)
    {
        return Camera.main.ScreenToWorldPoint(position);
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
        var layerPanel = DrawLayerPanel();
        for (int neuronIndex = 0; neuronIndex < neuralLayer.Neurons.Count; neuronIndex++)
        {
            var neuron = DrawNeuron(layerPanel);
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

    private VerticalLayoutGroup DrawLayerPanel()
    {
        var newLayerPanel = Instantiate(layerPanel);
        newLayerPanel.transform.SetParent(networkPanel.transform, false);
        return newLayerPanel;
    }

    private NeuronController DrawNeuron(VerticalLayoutGroup layerPanel)
    {
        var newNeuron = Instantiate(neuronPrefab);
        newNeuron.transform.SetParent(layerPanel.transform, false);
        newNeuron.MouseEnter += OnNeuronEnter;
        newNeuron.GetComponent<UnityEngine.UI.Image>().enabled = true;
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
