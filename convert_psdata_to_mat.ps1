# Set encoding to UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# Basic settings
$sourceDir = "W:\2024\0911"    # Working directory
$outputDir = "Z:\tmp"          # Output directory

# Define possible PicoScope 7 paths
$possiblePaths = @(
    "C:\Program Files\Pico Technology\PicoScope 7\PicoScope.exe",
    "C:\Program Files (x86)\Pico Technology\PicoScope 7\PicoScope.exe"
)

# Find PicoScope executable
$picoscopePath = $null
foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $picoscopePath = $path
        Write-Host "Found PicoScope 7 at: $picoscopePath"
        break
    }
}

if ($null -eq $picoscopePath) {
    Write-Host "Error: PicoScope 7 executable not found"
    exit 1
}

# Save current directory
$originalLocation = Get-Location
Write-Host "Current working directory: $originalLocation"

# 1. Check output directory
Write-Host "`nChecking output directory..."
if (!(Test-Path -Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir
    Write-Host "Created output directory: $outputDir"
}

# 2. Change to working directory
Write-Host "`nChanging to working directory..."
try {
    Set-Location -Path $sourceDir
    Write-Host "Successfully changed to: $(Get-Location)"
    
    # 3. Search for .psdata files
    Write-Host "`nSearching for .psdata files..."
    $psdataFiles = Get-ChildItem -Filter "*.psdata"
    $totalFiles = $psdataFiles.Count
    Write-Host "Found $totalFiles files"

    # 4. Convert each file
    foreach ($file in $psdataFiles) {
        Write-Host "`nProcessing: $($file.Name)"
        
        # Set output filename
        $outputFile = Join-Path $outputDir ($file.BaseName + ".mat")
        
        # Execute conversion
        try {
            # Create argument array for PicoScope 7
            $arguments = @(
                "--convert",
                $file.Name,
                "--format", "mat",
                "--output", $outputFile
            )
            
            Write-Host "Executing: `"$picoscopePath`" $arguments"
            
            # Execute command with argument array
            $result = & "$picoscopePath" $arguments
            
            # Check result
            if (Test-Path $outputFile) {
                Write-Host "Successfully converted: $($file.Name) -> $($file.BaseName).mat"
            } else {
                Write-Host "Error: Output file was not created"
                Write-Host "Command output: $result"
            }
        }
        catch {
            Write-Host "Error occurred: $_"
        }
    }
}
catch {
    Write-Host "Error occurred: $_"
}
finally {
    # 5. Return to original directory
    Write-Host "`nReturning to original directory..."
    Set-Location -Path $originalLocation
    Write-Host "Current working directory: $(Get-Location)"
}

Write-Host "`nConversion process completed" 