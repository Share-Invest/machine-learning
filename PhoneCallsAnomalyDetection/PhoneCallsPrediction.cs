using Microsoft.ML.Data;

namespace PhoneCallsAnomalyDetection;

class PhoneCallsPrediction
{
    [VectorType(7)]
    public double[]? Prediction
    {
        get; set;
    }
}