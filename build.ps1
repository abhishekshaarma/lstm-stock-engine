$requiredFiles = @("main.cpp", "lstm.cpp", "Utils.cpp")
foreach ($file in $requiredFiles) {
    if (!(Test-Path $file)) {
        Write-Host "Missing file: $file" 
        exit 1
    }
}

if (!(Test-Path "data/stock.csv")) {
    Write-Host "Warning: data/stock.csv not found!" 
}

Write-Host "Files found:" 
Get-ChildItem -Name "*.cpp", "*.h" | ForEach-Object { Write-Host "   $_"  }

Write-Host "Compiling..." 

$command = "g++ -std=c++17 -I`"./include`" -I`"./include/eigen-3.4.0`" -O2 main.cpp lstm.cpp Utils.cpp -o bin/lstm.exe"

Write-Host "Command: $command" 
Invoke-Expression $command

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" 
    Write-Host "Running LSTM model..." 
    Write-Host "This will take several minutes..." 
    Write-Host ""
    

    & "./bin/lstm.exe"
} else {
    Write-Host "Build failed!" 
}