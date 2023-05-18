using Microsoft.ML.Data;

namespace TextClassificationTF;

class MovieReviewSentimentPrediction
{
    [VectorType(2)]
    public float[]? Prediction
    {
        get; set;
    }
}