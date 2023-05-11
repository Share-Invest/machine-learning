using Microsoft.ML.Data;

namespace IrisFlowerClustering;

class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId
    {
        get; set;
    }
    [ColumnName("Score")]
    public float[]? Distances
    {
        get; set;
    }
}