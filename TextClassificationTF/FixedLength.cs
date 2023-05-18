using Microsoft.ML.Data;

namespace TextClassificationTF;

class FixedLength
{
    [VectorType(Config.FeatureLength)]
    public int[]? Features
    {
        get; set;
    }
}