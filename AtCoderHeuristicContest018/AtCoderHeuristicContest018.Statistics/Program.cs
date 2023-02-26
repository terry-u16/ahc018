using AtCoderHeuristicContest018.Statistics;
using System.Text.Json;

ConsoleApp.Run(args, async ([Option("d")] string directoryPath, [Option("j")] string jsonPath) =>
{
    var directory = new DirectoryInfo(directoryPath);

    var minScoreDict = new Dictionary<int, int>();

    foreach (var file in directory.EnumerateFiles("*.json"))
    {
        using var stream = file.OpenRead();
        var statistics = await JsonSerializer.DeserializeAsync<Statistics>(stream);

        if (statistics is null)
        {
            continue;
        }

        foreach (var result in statistics.Results)
        {
            if (result.Score == 0)
            {
                continue;
            }

            if (!minScoreDict.TryGetValue(result.Seed, out var score) || score > result.Score)
            {
                minScoreDict[result.Seed] = result.Score;
            }
        }
    }

    var total = 0.0;
    using var targetStream = new FileStream(jsonPath, FileMode.Open, FileAccess.Read);
    var targetStats = await JsonSerializer.DeserializeAsync<Statistics>(targetStream);

    foreach (var result in targetStats!.Results)
    {
        if (result.Score == 0)
        {
            continue;
        }

        var relativeScore = (double)minScoreDict[result.Seed] / result.Score;
        total += relativeScore;
    }

    total /= targetStats.Results.Length;
    Console.WriteLine($"seed count : {targetStats.Results.Length}");
    Console.WriteLine($"total score: {total:0.0000}");
});

