using System.Drawing;

namespace ObjectDetection.YoloParser;

class YoloBoundingBox
{
    public BoundingBoxDimensions? Dimensions
    {
        get; set;
    }
    public string? Label
    {
        get; set;
    }
    public float Confidence
    {
        get; set;
    }
    public RectangleF Rect
    {
        get => Dimensions != null ?

            new(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height) :

            RectangleF.Empty;
    }
    public Color BoxColor
    {
        get; set;
    }
}