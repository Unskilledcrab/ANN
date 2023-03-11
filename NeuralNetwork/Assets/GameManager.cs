using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    private Material defaultMaterial;
    private NeuralNetwork network;
    private int maxNeuronsInLayer;
    private List<List<GameObject>> networkNeurons;
    private float offset = 2.5f;
    private float trainingInterval = 0.05f;

    public int InputNeurons { get; set; } = 4;
    public int OutputNeurons { get; set; } = 3;


    // Start is called before the first frame update
    void Start()
    {
        var plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
        plane.SetActive(false);
        defaultMaterial = plane.GetComponent<MeshRenderer>().sharedMaterial;

        network = NeuralNetworkBuilder
            .CreateNetwork()
            .WithSettings(0.5, new PowerDifferenceErrorFunction())
            .WithInputLayer(InputNeurons)
            .WithHiddenLayer(new LayerConfiguration { Neurons = 5 })
            .WithHiddenLayer(new LayerConfiguration { Neurons = 6 })
            .WithHiddenLayer(new LayerConfiguration { Neurons = 4 })
            .WithOutputLayer(new LayerConfiguration { Neurons = OutputNeurons, ActivationFunction = new SigmoidActivationFunction() });

        maxNeuronsInLayer = network.Layers.Select(l => l.Neurons.Count).Max();
        networkNeurons = DrawNeuralNetwork(network);
        DrawConnections();
        InvokeRepeating("TrainNetwork", trainingInterval, trainingInterval);
    }

    private void TrainNetwork()
    {
        var trainingSets = FakeData.HardCodedSets();
        network.Train(trainingSets, 1);
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
                    synapse.LineRenderer = lineRenderer;
                    var currentNeuron = network.Layers[currentLayerIndex].Neurons[currentNeuronIndex].NeuronObject;
                    var nextLayerNeuron = network.Layers[nextLayerIndex].Neurons[nextLayerNeuronIndex].NeuronObject;

                    lineRenderer.SetPosition(0, currentNeuron.transform.position);
                    lineRenderer.SetPosition(1, nextLayerNeuron.transform.position);
                }
            }
        }
    }

    private LineRenderer CreateLineRenderer()
    {
        var gameObject = new GameObject();
        gameObject.AddComponent<LineRenderer>();
        var lineRenderer = gameObject.GetComponent<LineRenderer>();
        lineRenderer.material = new Material(defaultMaterial);
        lineRenderer.material.color = Color.gray;
        lineRenderer.positionCount = 2;
        lineRenderer.SetWidth(0.1f, 0.1f);
        return lineRenderer;
    }

    private List<List<GameObject>> DrawNeuralNetwork(NeuralNetwork network)
    {
        var networkNeurons = new List<List<GameObject>>();
        for (int layerIndex = 0; layerIndex < network.Layers.Count; layerIndex++)
        {
            var neurons = DrawLayer(layerIndex, network.Layers[layerIndex]);
            networkNeurons.Add(neurons);
        }
        return networkNeurons;
    }

    private List<GameObject> DrawLayer(int layerIndex, NeuralLayer neuralLayer)
    {
        var neurons = new List<GameObject>();
        for (int neuronIndex = 0; neuronIndex < neuralLayer.Neurons.Count; neuronIndex++)
        {
            var neuron = DrawNeuron(layerIndex, neuronIndex, neuralLayer.Neurons[neuronIndex]);
            neuralLayer.Neurons[neuronIndex].NeuronObject = neuron;
            neurons.Add(neuron);
        }
        var totalOffset = maxNeuronsInLayer - neurons.Count;
        var actualOffset = totalOffset / 2.0f;
        foreach (var neuron in neurons)
        {
            neuron.transform.Translate(Vector3.up * actualOffset * offset);
        }
        return neurons;
    }

    private GameObject DrawNeuron(int layerIndex, int neuronIndex, Neuron neuron)
    {
        var unityNeuron = CreateNeuron();
        unityNeuron.GetComponent<Renderer>().material.color = Color.blue;
        unityNeuron.transform.position = new Vector3(layerIndex * offset, neuronIndex * offset, 0);
        return unityNeuron;
    }

    GameObject CreateNeuron()
    {
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        return sphere;
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
            var nextLayerIndex = currentLayerIndex + 1;
            for (int currentNeuronIndex = 0; currentNeuronIndex < networkNeurons[currentLayerIndex].Count; currentNeuronIndex++)
            {
                for (int nextLayerNeuronIndex = 0; nextLayerNeuronIndex < networkNeurons[nextLayerIndex].Count; nextLayerNeuronIndex++)
                {
                    var currentNeuron = network.Layers[currentLayerIndex].Neurons[currentNeuronIndex];
                    var synapse = currentNeuron.OutputSynapses[nextLayerNeuronIndex];
                    var lineRenderer = synapse.LineRenderer;
                    var currentNeuronObject = currentNeuron.NeuronObject;
                    var nextLayerNeuronObject = network.Layers[nextLayerIndex].Neurons[nextLayerNeuronIndex].NeuronObject;

                    lineRenderer.SetPosition(0, currentNeuronObject.transform.position);
                    lineRenderer.SetPosition(1, nextLayerNeuronObject.transform.position);
                }
            }
        }
    }
}
