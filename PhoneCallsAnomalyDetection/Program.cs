using Microsoft.ML;
using Microsoft.ML.TimeSeries;

using PhoneCallsAnomalyDetection;

int DetectPeriod(MLContext context, IDataView phoneCalls)
{
    var period = context.AnomalyDetection.DetectSeasonality(phoneCalls, nameof(PhoneCallsData.Value));

    Console.WriteLine("Period of the series is: {0}.", period);

    return period;
}
void DetectAnomaly(MLContext context, IDataView phoneCalls, int period)
{
    var options = new SrCnnEntireAnomalyDetectorOptions
    {
        Threshold = 0.3,
        Sensitivity = 64,
        DetectMode = SrCnnDetectMode.AnomalyAndMargin,
        Period = period
    };
    var outputDataView =

        context.AnomalyDetection.DetectEntireAnomalyBySrCnn(phoneCalls,
                                                            nameof(PhoneCallsPrediction.Prediction),
                                                            nameof(PhoneCallsData.Value),
                                                            options);

    var predictions = context.Data.CreateEnumerable<PhoneCallsPrediction>(outputDataView, true);

    Console.WriteLine("Index,Data,Anomaly,AnomalyScore,Mag,ExpectedValue,BoundaryUnit,UpperBoundary,LowerBoundary");

    var index = 0;

    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            string output;

            if (p.Prediction[0] == 1)
            {
                output = "{0},{1},{2},{3},{4},  <-- alert is on! detected anomaly";
            }
            else
            {
                output = "{0},{1},{2},{3},{4}";
            }
            Console.WriteLine(output, index, p.Prediction[0], p.Prediction[3], p.Prediction[5], p.Prediction[6]);
        }
        index++;
    }
}
var _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "phone-calls.csv");

var context = new MLContext();

var dataView = context.Data.LoadFromTextFile<PhoneCallsData>(_dataPath,
                                                             hasHeader: true,
                                                             separatorChar: ',');

var period = DetectPeriod(context, dataView);

DetectAnomaly(context, dataView, period);
