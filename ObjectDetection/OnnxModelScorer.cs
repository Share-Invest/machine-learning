using Microsoft.ML;
using Microsoft.ML.Data;

using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;

namespace ObjectDetection;

class OnnxModelScorer
{
    internal IEnumerable<float[]> Score(IDataView data)
    {
        var model = LoadModel(modelLocation);

        return PredictDataUsingModel(data, model);
    }
    internal OnnxModelScorer(string imagesFolder, string modelLocation, MLContext context)
    {
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        this.context = context;
    }
    ITransformer LoadModel(string modelLocation)
    {
        Console.WriteLine("Read model");
        Console.WriteLine($"Model location: {modelLocation}");
        Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

        var data = context.Data.LoadFromEnumerable(new List<ImageNetData>());

        var pipeline =

            context.Transforms.LoadImages(image,
                                          string.Empty,
                                          inputColumnName: nameof(ImageNetData.ImagePath))

            .Append(context.Transforms.ResizeImages(image,
                                                    ImageNetSettings.imageWidth,
                                                    ImageNetSettings.imageHeight,
                                                    inputColumnName: image))

            .Append(context.Transforms.ExtractPixels(image))

            .Append(context.Transforms.ApplyOnnxModel(new[]
                                                      {
                                                          TinyYoloModelSettings.ModelOutput
                                                      },
                                                      new[]
                                                      {
                                                          TinyYoloModelSettings.ModelInput
                                                      },
                                                      modelLocation));

        return pipeline.Fit(data);
    }
    IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        Console.WriteLine($"Images location: {imagesFolder}");
        Console.WriteLine("");
        Console.WriteLine("=====Identify the objects in the images=====");
        Console.WriteLine("");

        var scoredData = model.Transform(testData);

        return scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);
    }
    readonly string imagesFolder;
    readonly string modelLocation;
    readonly MLContext context;
    /*
    readonly IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();
    */
    const string image = "image";
}