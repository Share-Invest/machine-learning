using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures;

class ImageNetData
{
    [LoadColumn(0)]
    public string? ImagePath
    {
        get; set;
    }
    [LoadColumn(1)]
    public string? Label
    {
        get; set;
    }
    public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder) =>

        Directory.GetFiles(imageFolder)
                 .Where(fp => Path.GetExtension(fp) != ".md")
                 .Select(fp => new ImageNetData
                 {
                     ImagePath = fp,
                     Label = Path.GetFileName(fp),
                 });
}