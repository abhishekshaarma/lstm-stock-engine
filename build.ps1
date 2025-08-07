# PowerShell build script for LSTM Multi-Factor Investment Model
# Requires Visual Studio Build Tools or MinGW-w64

$ErrorActionPreference = "Stop"

# Configuration
$CXX = "g++"
$CXXFLAGS = "-std=c++17 -Wall -O2 -fopenmp"
$INCLUDES = "-Iinclude -Iinclude/eigen-3.4.0"
$LIBS = "-fopenmp"

# Directories
$OBJDIR = "obj"
$BINDIR = "bin"

# Create directories
if (!(Test-Path $OBJDIR)) { New-Item -ItemType Directory -Path $OBJDIR }
if (!(Test-Path $BINDIR)) { New-Item -ItemType Directory -Path $BINDIR }

# Source files
$SOURCES = @("lstm.cpp", "main.cpp", "Utils.cpp")
$OBJECTS = @()

# Compile source files
foreach ($source in $SOURCES) {
    $obj = "$OBJDIR\$([System.IO.Path]::GetFileNameWithoutExtension($source)).o"
    $OBJECTS += $obj
    
    Write-Host "Compiling $source..."
    & $CXX $CXXFLAGS $INCLUDES -c $source -o $obj
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Compilation failed for $source"
        exit 1
    }
}

# Link executable
Write-Host "Linking executable..."
& $CXX $OBJECTS -o "$BINDIR/lstm.exe" $LIBS
if ($LASTEXITCODE -ne 0) {
    Write-Error "Linking failed"
    exit 1
}

Write-Host "Build completed successfully!"
Write-Host "Executable: $BINDIR/lstm.exe"