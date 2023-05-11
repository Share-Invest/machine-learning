using Microsoft.ML;

using PredictPricesUsingRegression;

const string label = "Label";
const string encode = "Encoded";
const string feature = "Features";
const string score = "Score";
const string data = "Data";

string _trainDataPath = Path.Combine(Environment.CurrentDirectory, data, "taxi-fare-train.csv"),
       _testDataPath = Path.Combine(Environment.CurrentDirectory, data, "taxi-fare-test.csv"),
       _modelPath = Path.Combine(Environment.CurrentDirectory, data, "Model.zip");

var context = new MLContext(seed: 0);

var model = Train(context, _trainDataPath);

Evaluate(context, model);

TestSinglePrediction(context, model);

void TestSinglePrediction(MLContext context, ITransformer model)
{
    var textTripSample = new TaxiTrip
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD"
    };
    var predictionFunction = context.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

    var prediction = predictionFunction.Predict(textTripSample);

    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine($"**********************************************************************");
}
void Evaluate(MLContext context, ITransformer model)
{
    var dataView = context.Data.LoadFromTextFile<TaxiTrip>(_testDataPath, hasHeader: true, separatorChar: (char)44);

    var predictions = model.Transform(dataView);

    var metrics = context.Regression.Evaluate(predictions, label, score);

    Console.WriteLine($"*************************************************");
    Console.WriteLine($"*       Model quality metrics evaluation         ");
    Console.WriteLine($"*------------------------------------------------");
    Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
    Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
}
ITransformer Train(MLContext context, string dataPath)
{
    string vendorId = string.Concat(nameof(TaxiTrip.VendorId), encode),
           rate = string.Concat(nameof(TaxiTrip.RateCode), encode),
           payment = string.Concat(nameof(TaxiTrip.PaymentType), encode);

    var dataView = context.Data.LoadFromTextFile<TaxiTrip>(dataPath, hasHeader: true, separatorChar: (char)44);

    var pipeline = context.Transforms.CopyColumns(label, nameof(TaxiTrip.FareAmount))

        .Append(context.Transforms.Categorical.OneHotEncoding(vendorId,
                                                              inputColumnName: nameof(TaxiTrip.VendorId)))
        .Append(context.Transforms.Categorical.OneHotEncoding(rate,
                                                              inputColumnName: nameof(TaxiTrip.RateCode)))
        .Append(context.Transforms.Categorical.OneHotEncoding(payment,
                                                              inputColumnName: nameof(TaxiTrip.PaymentType)))
        .Append(context.Transforms.Concatenate(feature,
                                               vendorId,
                                               rate,
                                               nameof(TaxiTrip.PassengerCount),
                                               nameof(TaxiTrip.TripDistance),
                                               payment))
        .Append(context.Regression.Trainers.FastTree());

    var model = pipeline.Fit(dataView);

    context.Model.Save(model, dataView.Schema, _modelPath);

    return model;
}