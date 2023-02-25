param(
    [Parameter(mandatory)]
    [string]
    $jsonPath
)

$directoryPath = "data\results"
dotnet run -c Release --project "AtCoderHeuristicContest018\AtCoderHeuristicContest018.Statistics\AtCoderHeuristicContest018.Statistics.csproj" -- -d $directoryPath -j $jsonPath
