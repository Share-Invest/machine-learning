using GitHubIssueClassification;

using Microsoft.ML;

const string label = "Label";
const string title = "TitleFeaturized";
const string description = "DescriptionFeaturized";
const string features = "Features";

string _appPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0])!;
string _trainDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
string _testDataPath = Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
string _modelPath = Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

MLContext _context;
PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
ITransformer _trainedModel;
IDataView _trainingDataView;

IEstimator<ITransformer> ProcessData() =>

    _context.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(GitHubIssue.Area), outputColumnName: label)
        .Append(_context.Transforms.Text.FeaturizeText(inputColumnName: nameof(GitHubIssue.Title), outputColumnName: title))
        .Append(_context.Transforms.Text.FeaturizeText(inputColumnName: nameof(GitHubIssue.Description), outputColumnName: description))
        .Append(_context.Transforms.Concatenate(features, title, description))
        .AppendCacheCheckpoint(_context);

IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
{
    var trainingPipeline =

        pipeline.Append(_context.MulticlassClassification.Trainers.SdcaMaximumEntropy(label, features))
                .Append(_context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

    _trainedModel = trainingPipeline.Fit(trainingDataView);

    _predEngine = _context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

    var prediction = _predEngine.Predict(new GitHubIssue
    {
        Title = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    });
    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

    return trainingPipeline;
}
void PredictIssue()
{
    ITransformer loadedModel = _context.Model.Load(_modelPath, out var _);

    _predEngine = _context.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);

    var prediction = _predEngine.Predict(new GitHubIssue
    {
        Title = "Entity Framework crashes",
        Description = "When connecting to the database, EF is crashing"
    });
    Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
}
void SaveModelAsFile(MLContext context, DataViewSchema trainingDataViewSchema, ITransformer model) =>

    context.Model.Save(model, trainingDataViewSchema, _modelPath);

void Evaluate(DataViewSchema trainingDataViewSchema)
{
    var testDataView = _context.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);

    var testMetrics = _context.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

    Console.WriteLine($"*************************************************************************************************************");
    Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
    Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
    Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
    Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
    Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
    Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
    Console.WriteLine($"*************************************************************************************************************");

    SaveModelAsFile(_context, trainingDataViewSchema, _trainedModel);
}
_context = new MLContext();

_trainingDataView = _context.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true);

var pipeline = ProcessData();

BuildAndTrainModel(_trainingDataView, pipeline);

Evaluate(_trainingDataView.Schema);

PredictIssue();
