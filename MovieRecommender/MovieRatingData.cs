using Microsoft.ML.Data;

namespace MovieRecommender;

class MovieRating
{
    [LoadColumn(0)]
    public float UserId
    {
        get; set;
    }
    [LoadColumn(1)]
    public float MovieId
    {
        get; set;
    }
    [LoadColumn(2)]
    public float Label
    {
        get; set;
    }
}