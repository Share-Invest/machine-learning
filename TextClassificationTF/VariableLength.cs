using Microsoft.ML.Data;

namespace TextClassificationTF;

class VariableLength
{
    [VectorType]
    public int[]? VariableLengthFeatures
    {
        get; set;
    }
}