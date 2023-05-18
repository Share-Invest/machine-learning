using BikeDemandForecasting;

using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

using System.Data;
using System.Data.SqlClient;

void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext context)
{
    var forecast = forecaster.Predict();

    var forecastOutput =

        context.Data.CreateEnumerable<ModelInput>(testData, false)
                    .Take(horizon)
                    .Select((rental, index) =>
                    {
                        if (forecast.LowerBoundRentals != null)
                        {
                            var rentalDate = rental.RentalDate.ToShortDateString();
                            var actualRentals = rental.TotalRentals;
                            var lowerEstimate = Math.Max(0, forecast.LowerBoundRentals[index]);
                            var estimate = forecast.ForecastedRentals?[index];
                            var upperEstimate = forecast.UpperBoundRentals?[index];

                            return $"Date: {rentalDate}\nActual Rentals: {actualRentals}\nLower Estimate: {lowerEstimate}\nForecast: {estimate}\nUpper Estimate: {upperEstimate}\n";
                        }
                        return string.Empty;
                    });
    Console.WriteLine("Rental Forecast");
    Console.WriteLine("---------------------");

    foreach (var prediction in forecastOutput)
    {
        Console.WriteLine(prediction);
    }
}
var rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
var connectionString = new ConfigurationBuilder().SetBasePath(rootDir)
                                                 .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                                                 .Build()
                                                 .GetConnectionString("DBConnection");
var modelPath = Path.Combine(rootDir, "MLModel.zip");
var dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");

var context = new MLContext();

DatabaseLoader loader = context.Data.CreateDatabaseLoader<ModelInput>();

string query = "SELECT RentalDate, CAST(Year as REAL) as Year, CAST(TotalRentals as REAL) as TotalRentals FROM Rentals";

var dbSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, query);

var dataView = loader.Load(dbSource);

var firstYearData = context.Data.FilterRowsByColumn(dataView, nameof(ModelInput.Year), upperBound: 1);
var secondYearData = context.Data.FilterRowsByColumn(dataView, nameof(ModelInput.Year), lowerBound: 1);

var forecastingPipeline = context.Forecasting.ForecastBySsa(nameof(ModelOutput.ForecastedRentals),
                                                            nameof(ModelInput.TotalRentals),
                                                            7,
                                                            30,
                                                            365,
                                                            7,
                                                            confidenceLevel: 0.95f,
                                                            confidenceLowerBoundColumn: nameof(ModelOutput.LowerBoundRentals),
                                                            confidenceUpperBoundColumn: nameof(ModelOutput.UpperBoundRentals));

var forecaster = forecastingPipeline.Fit(firstYearData);

Evaluate(secondYearData, forecaster, context);

var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(context);

forecastEngine.CheckPoint(context, modelPath);

Forecast(secondYearData, 7, forecastEngine, context);

void Evaluate(IDataView testData, ITransformer model, MLContext context)
{
    var predictions = model.Transform(testData);

    var actual = context.Data.CreateEnumerable<ModelInput>(testData, true).Select(o => o.TotalRentals);

    var forecast = context.Data.CreateEnumerable<ModelOutput>(predictions, true).Select(p => p.ForecastedRentals?[0]);

    var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

    var mae = metrics.Average(error => Math.Abs(error ?? 0));
    var rmse = Math.Sqrt(metrics.Average(error => Math.Pow(error ?? 1, 2)));

    Console.WriteLine("Evaluation Metrics");
    Console.WriteLine("---------------------");
    Console.WriteLine($"Mean Absolute Error: {mae:F3}");
    Console.WriteLine($"Root Mean Squared Error: {rmse:F3}\n");
}