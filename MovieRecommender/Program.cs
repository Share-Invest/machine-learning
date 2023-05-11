using Microsoft.ML;
using Microsoft.ML.Trainers;

using MovieRecommender;

const string data = "Data";
const string user = "userIdEncoded";
const string movie = "movieIdEncoded";

var context = new MLContext();

(IDataView training, IDataView test) LoadData(MLContext context)
{
    var trainingDataPath = Path.Combine(Environment.CurrentDirectory, data, "recommendation-ratings-train.csv");
    var testDataPath = Path.Combine(Environment.CurrentDirectory, data, "recommendation-ratings-test.csv");

    IDataView trainingDataView = context.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
    IDataView testDataView = context.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

    return (trainingDataView, testDataView);
}
ITransformer BuildAndTrainModel(MLContext context, IDataView trainingDataView)
{
    IEstimator<ITransformer> estimator = context.Transforms.Conversion.MapValueToKey(outputColumnName: user, inputColumnName: nameof(MovieRating.UserId))
        .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: movie, inputColumnName: nameof(MovieRating.MovieId)));

    var options = new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = user,
        MatrixRowIndexColumnName = movie,
        LabelColumnName = nameof(MovieRating.Label),
        NumberOfIterations = 25,
        ApproximationRank = 100
    };
    var trainerEstimator = estimator.Append(context.Recommendation().Trainers.MatrixFactorization(options));

    Console.WriteLine("=============== Training the model ===============");

    return trainerEstimator.Fit(trainingDataView);
}
(IDataView trainingDataView, IDataView testDataView) = LoadData(context);

var model = BuildAndTrainModel(context, trainingDataView);

EvaluateModel(context, testDataView, model);

UseModelForSinglePrediction(context, model);

SaveModel(context, trainingDataView.Schema, model);

void EvaluateModel(MLContext context, IDataView testDataView, ITransformer model)
{
    Console.WriteLine("=============== Evaluating the model ===============");

    var prediction = model.Transform(testDataView);

    var metrics = context.Regression.Evaluate(prediction,
                                              labelColumnName: nameof(MovieRating.Label),
                                              scoreColumnName: nameof(MovieRatingPrediction.Score));

    Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
    Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
}
void UseModelForSinglePrediction(MLContext context, ITransformer model)
{
    Console.WriteLine("=============== Making a prediction ===============");

    var predictionEngine = context.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

    var testInput = new MovieRating
    {
        UserId = 6,
        MovieId = 10
    };
    var movieRatingPrediction = predictionEngine.Predict(testInput);

    if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
    {
        Console.WriteLine("Movie " + testInput.MovieId + " is recommended for user " + testInput.UserId);
    }
    else
    {
        Console.WriteLine("Movie " + testInput.MovieId + " is not recommended for user " + testInput.UserId);
    }
}
void SaveModel(MLContext context, DataViewSchema trainingDataViewSchema, ITransformer model)
{
    var modelPath = Path.Combine(Environment.CurrentDirectory, data, "MovieRecommenderModel.zip");

    Console.WriteLine("=============== Saving the model to a file ===============");

    context.Model.Save(model, trainingDataViewSchema, modelPath);
}