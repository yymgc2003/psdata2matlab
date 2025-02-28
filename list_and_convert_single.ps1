# Directory settings
$TARGET_DIR = "W:\2023\Ultrasound\012"  # Target directory with .psdata files
$OUTPUT_DIR = "Z:\tmp"                   # Output directory for .mat files
$TIMEOUT_SECONDS = 30                    # Timeout duration in seconds

# Save current directory
$original_location = Get-Location

Write-Output "=== Environment Info ==="
Write-Output "Current directory: $original_location"
Write-Output "Target directory: $TARGET_DIR"
Write-Output "Output directory: $OUTPUT_DIR"
Write-Output "Timeout: $TIMEOUT_SECONDS seconds"

# Change to target directory
Write-Output "`n=== Changing Directory ==="
try {
    Set-Location -Path $TARGET_DIR
    Write-Output "Successfully changed to: $(Get-Location)"
} catch {
    Write-Output "Error: Could not change to target directory: $_"
    exit 1
}

# List .psdata files
Write-Output "`n=== File List ==="
$psdata_files = Get-ChildItem -Filter "*.psdata" | Select-Object Name, Length, LastWriteTime
$total_files = $psdata_files.Count

Write-Output "Found $total_files .psdata files:"
$psdata_files | Format-Table -AutoSize

# Let user select a file
Write-Output "`n=== File Selection ==="
Write-Output "Enter the name of the file to process (example: s0_g1_l2_t1.psdata):"
$selected_file = Read-Host "Filename"

# Check if file exists
if (!(Test-Path $selected_file)) {
    Write-Output "Error: Selected file does not exist: $selected_file"
    Set-Location -Path $original_location
    exit 1
}

# Generate output filename
$output_file = Join-Path $OUTPUT_DIR ([System.IO.Path]::GetFileNameWithoutExtension($selected_file) + ".mat")

# Check file integrity
Write-Output "`n=== Checking File Integrity ==="
Write-Output "Attempting to verify file: $selected_file"

try {
    # Start file check job
    $checkJob = Start-Job -ScriptBlock {
        param($file)
        # Try to open the file with picoscope in info mode
        picoscope /c $file /i
    } -ArgumentList $selected_file

    # Wait for the check with timeout
    $checkCompleted = Wait-Job -Job $checkJob -Timeout 10
    
    if ($checkCompleted -eq $null) {
        Write-Output "Warning: File integrity check timed out - file might be corrupted"
        Stop-Job -Job $checkJob
        Remove-Job -Job $checkJob -Force
        
        Write-Output "Do you want to proceed anyway? (Y/N):"
        $proceedAnyway = Read-Host
        if ($proceedAnyway -ne "Y") {
            Write-Output "Operation cancelled"
            Set-Location -Path $original_location
            exit 0
        }
    } else {
        $checkResult = Receive-Job -Job $checkJob
        Remove-Job -Job $checkJob
        
        if ($LASTEXITCODE -ne 0) {
            Write-Output "Warning: File integrity check failed - file might be corrupted"
            Write-Output "Do you want to proceed anyway? (Y/N):"
            $proceedAnyway = Read-Host
            if ($proceedAnyway -ne "Y") {
                Write-Output "Operation cancelled"
                Set-Location -Path $original_location
                exit 0
            }
        } else {
            Write-Output "File integrity check passed"
        }
    }
} catch {
    Write-Output "Error during file integrity check: $_"
    Write-Output "Do you want to proceed anyway? (Y/N):"
    $proceedAnyway = Read-Host
    if ($proceedAnyway -ne "Y") {
        Write-Output "Operation cancelled"
        Set-Location -Path $original_location
        exit 0
    }
}

# Execute conversion
Write-Output "`n=== Executing Conversion ==="
Write-Output "Input file: $selected_file"
Write-Output "Output file: $output_file"
Write-Output "Command to execute: picoscope /c $selected_file /f mat /d $output_file /q /b 1"

Write-Output "`nExecute this command? (Y/N):"
$confirm = Read-Host
if ($confirm -ne "Y") {
    Write-Output "Operation cancelled"
    Set-Location -Path $original_location
    exit 0
}

try {
    # Start conversion job
    $conversionJob = Start-Job -ScriptBlock {
        param($file, $output)
        picoscope /c $file /f mat /d $output /q /b 1
        $LASTEXITCODE
    } -ArgumentList $selected_file, $output_file

    Write-Output "Conversion started. Waiting for completion..."
    $completed = Wait-Job -Job $conversionJob -Timeout $TIMEOUT_SECONDS

    if ($completed -eq $null) {
        Write-Output "`nWarning: Conversion timed out after $TIMEOUT_SECONDS seconds"
        Write-Output "The file might be corrupted or too large to process"
        Stop-Job -Job $conversionJob
        Remove-Job -Job $conversionJob -Force
        Write-Output "Operation cancelled due to timeout"
    } else {
        $exitCode = Receive-Job -Job $conversionJob
        Remove-Job -Job $conversionJob
        
        Write-Output "`n=== Results ==="
        Write-Output "Exit code: $exitCode"
        
        if ($exitCode -eq 0) {
            Write-Output "Success: $selected_file"
            if (Test-Path $output_file) {
                $fileInfo = Get-Item $output_file
                Write-Output "Created file size: $($fileInfo.Length) bytes"
            } else {
                Write-Output "Warning: Output file not found despite successful return code"
            }
        } else {
            Write-Output "Failed: $selected_file with exit code $exitCode"
        }
    }
} catch {
    Write-Output "Error occurred: $_"
}

# Return to original directory
Set-Location -Path $original_location
Write-Output "`nScript completed" 