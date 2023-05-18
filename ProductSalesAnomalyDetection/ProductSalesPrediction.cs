using Microsoft.ML.Data;

namespace ProductSalesAnomalyDetection;

class ProductSalesPrediction
{
    [VectorType(3)]
    public double[]? Prediction
    {
        get; set;
    }
}