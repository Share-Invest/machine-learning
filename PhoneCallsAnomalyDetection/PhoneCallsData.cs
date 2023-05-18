using Microsoft.ML.Data;

namespace PhoneCallsAnomalyDetection;

class PhoneCallsData
{
    [LoadColumn(0)]
    public string? TimeStamp
    {
        get; set;
    }
    [LoadColumn(1)]
    public double Value
    {
        get; set;
    }
}