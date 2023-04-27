using Microsoft.ML;
using Microsoft.ML.Data;

using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis;

class Program
{
    const string features = "Features";
    const string label = "Label";

    static void Main()
    {
        var context = new MLContext();

        TrainTestData splitDataView = LoadData(context);

        ITransformer model = BuildAndTrainModel(context, splitDataView.TrainSet);

        Evaluate(context, model, splitDataView.TestSet);

        UseModelWithSingleItem(context, model);

        UseModelWithBatchItems(context, model);
    }
    static void UseModelWithBatchItems(MLContext context, ITransformer model)
    {
        IDataView batchComments = context.Data.LoadFromEnumerable(new[]
        {
            new SentimentData
            {
                SentimentText = "This was a horrible meal."
            },
            new SentimentData
            {
                SentimentText = "I love this spaghetti."
            }
        });
        IDataView predictions = model.Transform(batchComments);

        foreach (var prediction in context.Data.CreateEnumerable<SentimentPrediction>(predictions, false))
        {
            Console.WriteLine($"Sentiment: {prediction.SentimentText}\nPrediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")}\nProbability: {prediction.Probability} ");
        }
    }
    static void UseModelWithSingleItem(MLContext context, ITransformer model)
    {
        var predictionFunction = context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

        var resultPrediction = predictionFunction.Predict(new SentimentData
        {
            SentimentText = "This was a very bad steak."
        });
        Console.WriteLine();
        Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

        Console.WriteLine();
        Console.WriteLine($"Sentiment: {resultPrediction.SentimentText}\nPrediction: {(Convert.ToBoolean(resultPrediction.Prediction) ? "Positive" : "Negative")}\nProbability: {resultPrediction.Probability} ");

        Console.WriteLine("=============== End of Predictions ===============");
        Console.WriteLine();
    }
    static void Evaluate(MLContext context, ITransformer model, IDataView splitTestSet)
    {
        IDataView predictions = model.Transform(splitTestSet);

        CalibratedBinaryClassificationMetrics metrics = context.BinaryClassification.Evaluate(predictions, label);

        Console.WriteLine();
        Console.WriteLine("Model quality metrics evaluation");
        Console.WriteLine("--------------------------------");
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        Console.WriteLine("=============== End of model evaluation ===============");
    }
    static ITransformer BuildAndTrainModel(MLContext context, IDataView splitTrainSet) =>

        context.Transforms.Text.FeaturizeText(features, inputColumnName: nameof(SentimentData.SentimentText))
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: label, featureColumnName: features))
            .Fit(splitTrainSet);

    static TrainTestData LoadData(MLContext context)
    {
        IDataView dataView = context.Data.LoadFromTextFile<SentimentData>(_dataPath, hasHeader: false);

        return context.Data.TrainTestSplit(dataView, testFraction: 0.2);
    }
    static readonly string _dataPath = Path.Combine(Directory.GetParent(Directory.GetParent(Directory.GetParent(Environment.CurrentDirectory)!.FullName)!.FullName)!.FullName, "Data", "yelp_labelled.txt");
}