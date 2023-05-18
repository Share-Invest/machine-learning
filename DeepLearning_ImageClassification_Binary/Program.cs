using DeepLearningImageClassificationBinary;

using Microsoft.ML;
using Microsoft.ML.Vision;

using static Microsoft.ML.DataOperationsCatalog;

IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
{
    foreach (var file in Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories))
    {
        if (Path.GetExtension(file) != ".jpg" && Path.GetExtension(file) != ".png")
            continue;

        var label = Path.GetFileName(file);

        if (useFolderNameAsLabel)
        {
            label = Directory.GetParent(file)?.Name;
        }
        else
            for (int i = 0; i < label.Length; i++)
            {
                if (char.IsLetter(label[i]) is false)
                {
                    label = label[..i];

                    break;
                }
            }
        yield return new ImageData
        {
            ImagePath = file,
            Label = label
        };
    }
}
var projectDirectory = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../"));
var workspaceRelativePath = Path.Combine(projectDirectory, "workspace");
var assetsRelativePath = Path.Combine(projectDirectory, "assets");

var context = new MLContext();

var images = LoadImagesFromDirectory(assetsRelativePath, useFolderNameAsLabel: true);

var imageData = context.Data.LoadFromEnumerable(images);

IDataView shuffledData = context.Data.ShuffleRows(imageData);

var preprocessingPipeline =

    context.Transforms.Conversion.MapValueToKey(nameof(ModelInput.LabelAsKey),
                                                inputColumnName: nameof(ImageData.Label))

    .Append(context.Transforms.LoadRawImageBytes(nameof(ModelInput.Image),
                                                 assetsRelativePath,
                                                 inputColumnName: nameof(ModelInput.ImagePath)));

IDataView preProcessedData =

    preprocessingPipeline.Fit(shuffledData).Transform(shuffledData);

TrainTestData trainSplit = context.Data.TrainTestSplit(preProcessedData, testFraction: 0.3);
TrainTestData validationTestSplit = context.Data.TrainTestSplit(trainSplit.TestSet);

IDataView trainSet = trainSplit.TrainSet,
          validationSet = validationTestSplit.TrainSet,
          testSet = validationTestSplit.TestSet;

var classifierOptions = new ImageClassificationTrainer.Options()
{
    FeatureColumnName = nameof(ModelInput.Image),
    LabelColumnName = nameof(ModelInput.LabelAsKey),
    ValidationSet = validationSet,
    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
    MetricsCallback = Console.WriteLine,
    TestOnTrainSet = false,
    ReuseTrainSetBottleneckCachedValues = true,
    ReuseValidationSetBottleneckCachedValues = true,
    WorkspacePath = workspaceRelativePath
};
var trainingPipeline =

    context.MulticlassClassification.Trainers.ImageClassification(classifierOptions)

    .Append(context.Transforms.Conversion.MapKeyToValue(nameof(ModelOutput.PredictedLabel)));

ITransformer trainedModel = trainingPipeline.Fit(trainSet);

ClassifySingleImage(testSet, trainedModel);

ClassifyImage(context, testSet, trainedModel);

void ClassifyImage(MLContext context, IDataView data, ITransformer trainedModel)
{
    IDataView predictionData = trainedModel.Transform(data);

    var predictions = context.Data.CreateEnumerable<ModelOutput>(predictionData, true).Take(10);

    Console.WriteLine("classifying multiple images");

    foreach (var prediction in predictions)
    {
        OutputPrediction(prediction);
    }
}
void ClassifySingleImage(IDataView data, ITransformer trainedModel)
{
    var predictionEngine =

        context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

    var image = context.Data.CreateEnumerable<ModelInput>(data, true).First();

    var prediction = predictionEngine.Predict(image);

    Console.WriteLine("classifying single image");

    OutputPrediction(prediction);
}
void OutputPrediction(ModelOutput prediction)
{
    var imageName = Path.GetFileName(prediction.ImagePath);

    Console.WriteLine($"Image: {imageName} | Actual Value: {prediction.Label} | Predicted Value: {prediction.PredictedLabel}");
}