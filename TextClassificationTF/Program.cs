using Microsoft.ML;
using Microsoft.ML.Data;

using TextClassificationTF;

static void PredictSentiment(MLContext context, ITransformer model)
{
    var review = new MovieReview()
    {
        ReviewText = "this film is really good"
    };
    var engine = context.Model.CreatePredictionEngine<MovieReview, MovieReviewSentimentPrediction>(model);

    var sentimentPrediction = engine.Predict(review);

    Console.WriteLine($"Number of classes: {sentimentPrediction.Prediction?.Length}");
    Console.WriteLine($"Is sentiment/review positive? {(sentimentPrediction.Prediction?[1] > 0.5 ? "Yes." : "No.")}");
}
const string words = "Words";
const string ids = "Ids";
const string tw = "TokenizedWords";
const string ps = "Prediction/Softmax";

var _modelPath = Path.Combine(Environment.CurrentDirectory, "sentiment_model");

var context = new MLContext();

var lookupMap = context.Data.LoadFromTextFile(Path.Combine(_modelPath, "imdb_word_index.csv"),
                                              columns: new[]
                                              {
                                                  new TextLoader.Column(words, DataKind.String, 0),
                                                  new TextLoader.Column(ids, DataKind.Int32, 1)
                                              },
                                              separatorChar: ',');

Action<VariableLength, FixedLength> ResizeFeaturesAction = (s, f) =>
{
    var features = s.VariableLengthFeatures;

    Array.Resize(ref features, Config.FeatureLength);

    f.Features = features;
};
var tensorFlowModel = context.Model.LoadTensorFlowModel(_modelPath);

var schema = tensorFlowModel.GetModelSchema();
Console.WriteLine(" =============== TensorFlow Model Schema =============== ");
var featuresType = (VectorDataViewType)schema[nameof(FixedLength.Features)].Type;
Console.WriteLine($"Name: {nameof(FixedLength.Features)}, Type: {featuresType.ItemType.RawType}, Size: ({featuresType.Dimensions[0]})");
var predictionType = (VectorDataViewType)schema[ps].Type;
Console.WriteLine($"Name: {ps}, Type: {predictionType.ItemType.RawType}, Size: ({predictionType.Dimensions[0]})");

var pipeline =

    context.Transforms.Text.TokenizeIntoWords(tw, nameof(MovieReview.ReviewText))

    .Append(context.Transforms.Conversion.MapValue(nameof(VariableLength.VariableLengthFeatures),
                                                   lookupMap,
                                                   lookupMap.Schema[words],
                                                   lookupMap.Schema[ids],
                                                   inputColumnName: tw))

    .Append(context.Transforms.CustomMapping(ResizeFeaturesAction, "Resize"))

    .Append(tensorFlowModel.ScoreTensorFlowModel(ps, nameof(FixedLength.Features)))

    .Append(context.Transforms.CopyColumns(nameof(MovieReviewSentimentPrediction.Prediction), ps));

var dataView = context.Data.LoadFromEnumerable(new List<MovieReview>());

var model = pipeline.Fit(dataView);

PredictSentiment(context, model);
