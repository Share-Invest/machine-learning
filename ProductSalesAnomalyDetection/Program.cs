using Microsoft.ML;

using ProductSalesAnomalyDetection;

IDataView CreateEmptyDataView(MLContext context)
{
    var enumerableData = new List<ProductSalesData>();

    return context.Data.LoadFromEnumerable(enumerableData);
}
void DetectSpike(MLContext context, int docsize, IDataView productSales)
{
    var iidSpikeEstimator = context.Transforms.DetectIidSpike(nameof(ProductSalesPrediction.Prediction),
                                                              nameof(ProductSalesData.NumSales),
                                                              95d,
                                                              docsize / 4);

    var iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(context));

    var transformedData = iidSpikeTransform.Transform(productSales);

    var predictions = context.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, false);

    Console.WriteLine("Alert\tScore\tP-Value");

    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- Spike detected";
            }
            Console.WriteLine(results);
        }
    }
}
void DetectChangePoint(MLContext context, int docSize, IDataView productSales)
{
    var iidChangePointEstimator = context.Transforms.DetectIidChangePoint(nameof(ProductSalesPrediction.Prediction),
                                                                          nameof(ProductSalesData.NumSales),
                                                                          95d,
                                                                          docSize / 4);

    var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(context));

    var transformedData = iidChangePointTransform.Transform(productSales);

    var predictions = context.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, false);

    Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");

    foreach (var p in predictions)
    {
        if (p.Prediction is not null)
        {
            var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

            if (p.Prediction[0] == 1)
            {
                results += " <-- alert is on, predicted changepoint";
            }
            Console.WriteLine(results);
        }
    }
}
const int _docsize = 36;

var _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "product-sales.csv");

var context = new MLContext();

var dataView = context.Data.LoadFromTextFile<ProductSalesData>(_dataPath,
                                                               hasHeader: true,
                                                               separatorChar: ',');

DetectSpike(context, _docsize, dataView);

DetectChangePoint(context, _docsize, dataView);
