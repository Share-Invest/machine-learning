using IrisFlowerClustering;

using Microsoft.ML;

const string featuresColumnName = "Features";

var _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");
var _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

var context = new MLContext(seed: 9);

IDataView dataView = context.Data.LoadFromTextFile<IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

var pipeline = context.Transforms.Concatenate(featuresColumnName, nameof(IrisData.SepalLength),
                                                                  nameof(IrisData.SepalWidth),
                                                                  nameof(IrisData.PetalLength),
                                                                  nameof(IrisData.PetalWidth))

                                 .Append(context.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

var model = pipeline.Fit(dataView);

using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
{
    context.Model.Save(model, dataView.Schema, fileStream);
}
var predictor = context.Model.CreatePredictionEngine<IrisData, ClusterPrediction>(model);

var prediction = predictor.Predict(TestIrisData.Setosa);

Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances ?? Array.Empty<float>())}");
