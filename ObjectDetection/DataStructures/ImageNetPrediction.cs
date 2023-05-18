using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures;

class ImageNetPrediction
{
    [ColumnName("grid")]
    public float[]? PredictedLabels
    {
        get; set;
    }
}