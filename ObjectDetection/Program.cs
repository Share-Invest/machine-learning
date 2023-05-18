using Microsoft.ML;

using ObjectDetection;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;

using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Runtime.Versioning;

string GetAbsulutePath(string relativePath)
{
    var _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

    var assemblyFolderPath = _dataRoot.Directory?.FullName;

    return Path.Combine(assemblyFolderPath ?? string.Empty, relativePath);
}
[SupportedOSPlatform("windows"),
 SuppressMessage("Microsoft.Usage", "CA1416")]
void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string? imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
{
    Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName ?? string.Empty));

    var originalImageHeight = image.Height;
    var originalImageWidth = image.Width;

    foreach (var box in filteredBoundingBoxes)
    {
        if (box.Dimensions == null)
        {
            continue;
        }
        var x = (uint)Math.Max(box.Dimensions.X, 0);
        var y = (uint)Math.Max(box.Dimensions.Y, 0);
        var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
        var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

        x = (uint)originalImageWidth * x / ImageNetSettings.imageWidth;
        y = (uint)originalImageHeight * y / ImageNetSettings.imageHeight;
        width = (uint)originalImageWidth * width / ImageNetSettings.imageWidth;
        height = (uint)originalImageHeight * height / ImageNetSettings.imageHeight;

        string text = $"{box.Label} ({box.Confidence * 100:0}%)";

        using (Graphics thumbnailGraphic = Graphics.FromImage(image))
        {
            thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
            thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
            thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

            using (Font drawFont = new("Arial", 12, FontStyle.Bold))
            {
                SizeF size = thumbnailGraphic.MeasureString(text, drawFont);

                using (SolidBrush fontBrush = new(Color.Black))
                {
                    Point atPoint = new((int)x, (int)y - (int)size.Height - 1);

                    using (Pen pen = new(box.BoxColor, 3.2f))
                    {
                        using (SolidBrush colorBrush = new(box.BoxColor))
                        {
                            thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                        }
                        thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                        thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                    }
                }
            }
        }
        if (Directory.Exists(outputImageLocation) is false)
        {
            Directory.CreateDirectory(outputImageLocation);
        }
        image.Save(Path.Combine(outputImageLocation, imageName ?? string.Empty));
    }
}
void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
{
    Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

    foreach (var box in boundingBoxes)
    {
        Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
    }

    Console.WriteLine("");
}
var assetsRelativePath = @"../../../assets";
var assetsPath = GetAbsulutePath(assetsRelativePath);
var modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
var imagesFolder = Path.Combine(assetsPath, "images");
var outputFolder = Path.Combine(assetsPath, "images", "output");

var context = new MLContext();

try
{
    IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);
    IDataView imageDataView = context.Data.LoadFromEnumerable(images);

    var modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, context);

    IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

    YoloOutputParser parser = new();

    var boundingBoxes =

        probabilities.Select(probability => parser.ParseOutputs(probability))
                     .Select(boxes => YoloOutputParser.FilterBoundingBoxes(boxes, 5, .5F));

    for (var i = 0; i < images.Count(); i++)
    {
        var imageFileName = images.ElementAt(i).Label;

        IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);

        DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);

        LogDetectedObjects(imageFileName ?? string.Empty, detectedObjects);
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}