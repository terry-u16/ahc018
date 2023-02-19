Write-Host "[Compile]"
cargo build --release
Move-Item ../target/release/ahc018.exe . -Force
Write-Host "[Run]"
$env:DURATION_MUL = "0.6"
dotnet marathon run-local