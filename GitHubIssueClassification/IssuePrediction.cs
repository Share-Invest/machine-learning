using Microsoft.ML.Data;

namespace GitHubIssueClassification;

class IssuePrediction
{
    [ColumnName("PredictedLabel")]
    public string? Area
    {
        get; set;
    }
}