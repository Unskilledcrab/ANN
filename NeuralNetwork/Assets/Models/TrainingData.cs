using System.Collections.Generic;
using System.Linq;

public class TrainingSet
{
    public List<double> Inputs { get; set; } = new();
    public List<double> ExpectedOutputs { get; set; } = new();
}

public static class FakeData
{
    public static List<TrainingSet> HardCodedSets()
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
}