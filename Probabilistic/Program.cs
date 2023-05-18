using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

var winnerData = new[] { 0, 0, 0, 1, 3, 4 };
var loserData = new[] { 1, 3, 4, 2, 1, 2 };

var game = new Microsoft.ML.Probabilistic.Models.Range(winnerData.Length);
var player = new Microsoft.ML.Probabilistic.Models.Range(winnerData.Concat(loserData).Max() + 1);
var playerSkills = Variable.Array<double>(player);

playerSkills[player] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(player);

var winners = Variable.Array<int>(game);
var losers = Variable.Array<int>(game);

using (Variable.ForEach(game))
{
    var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], 1.0);
    var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], 1.0);

    Variable.ConstrainTrue(winnerPerformance > loserPerformance);
}
winners.ObservedValue = winnerData;
losers.ObservedValue = loserData;

var inferenceEngine = new InferenceEngine();
var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);

var orderedPlayerSkills =

    inferredSkills.Select((s, i) => new { Player = i, Skill = s })
                  .OrderByDescending(ps => ps.Skill.GetMean());

foreach (var playerSkill in orderedPlayerSkills)
{
    Console.WriteLine($"Player {playerSkill.Player} skill: {playerSkill.Skill}");
}