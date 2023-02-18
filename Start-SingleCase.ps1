param(
    [Parameter(mandatory)]
    [int]
    $seed
)

$in = ".\data\in\{0:0000}.txt" -f $seed
Get-Content $in | .\tester.exe cargo run --release > out.txt
