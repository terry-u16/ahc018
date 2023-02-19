Write-Host "[Compile]"
expander-rs src/main.rs
dotnet marathon compile-rust
Write-Host "[Run]"
dotnet marathon run-cloud