using Microsoft.ML.Data;

namespace PredictPricesUsingRegression;

class TaxiTripFarePrediction
{
    [ColumnName("Score")]
    public float FareAmount
    {
        get; set;
    }
}