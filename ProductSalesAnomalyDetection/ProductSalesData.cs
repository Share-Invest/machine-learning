using Microsoft.ML.Data;

namespace ProductSalesAnomalyDetection;

class ProductSalesData
{
    [LoadColumn(0)]
    public string? Month
    {
        get; set;
    }
    [LoadColumn(1)]
    public float NumSales
    {
        get; set;
    }
}