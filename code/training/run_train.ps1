param(
    [Parameter(Mandatory = $true)]
    [string]$GeneratedDir,

    [string]$LlamaFactoryDir = "..\LLaMA-Factory",

    [ValidateSet("smoke", "336", "448")]
    [string]$Profile = "smoke",

    [string]$RunName = "",

    [ValidateSet("auto", "thought_action", "action_only")]
    [string]$AssistantFormat = "auto",

    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"

$TrainingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$CodeDir = Resolve-Path (Join-Path $TrainingDir "..")

if ([System.IO.Path]::IsPathRooted($LlamaFactoryDir)) {
    $LfRoot = Resolve-Path $LlamaFactoryDir
} else {
    $LfRoot = Resolve-Path (Join-Path $CodeDir $LlamaFactoryDir)
}

if ([System.IO.Path]::IsPathRooted($GeneratedDir)) {
    $GenRoot = Resolve-Path $GeneratedDir
} else {
    $GenRoot = Resolve-Path (Join-Path $CodeDir $GeneratedDir)
}

if ($Profile -eq "smoke") {
    $Config = Join-Path $TrainingDir "qwen2_5vl_lora_sft_smoke.yaml"
    $DefaultOutput = "saves/qwen2_5vl-7b/lora/uav_sft_smoke"
} elseif ($Profile -eq "448") {
    $Config = Join-Path $TrainingDir "qwen2_5vl_lora_sft_448.yaml"
    $DefaultOutput = "saves/qwen2_5vl-7b/lora/uav_sft_448"
} else {
    $Config = Join-Path $TrainingDir "qwen2_5vl_lora_sft_336.yaml"
    $DefaultOutput = "saves/qwen2_5vl-7b/lora/uav_sft_336"
}

if ([string]::IsNullOrWhiteSpace($RunName)) {
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunName = "run_${Profile}_${Timestamp}"
}
$OutputDir = "$DefaultOutput\$RunName"
$TmpConfig = Join-Path $LfRoot ".airnav_train_${Profile}_${RunName}.yaml"

if (-not $SkipPrepare) {
    python (Join-Path $TrainingDir "prepare_llamafactory_dataset.py") `
        --generated_dir $GenRoot `
        --llamafactory_dir $LfRoot `
        --assistant_format $AssistantFormat
}

python (Join-Path $TrainingDir "materialize_train_config.py") `
    --template $Config `
    --output_dir $OutputDir `
    --destination $TmpConfig

Push-Location $LfRoot
try {
    Write-Host "[train] profile=$Profile run_name=$RunName"
    Write-Host "[train] output_dir=$OutputDir"
    llamafactory-cli train $TmpConfig
} finally {
    Pop-Location
    if (Test-Path -LiteralPath $TmpConfig) {
        Remove-Item -LiteralPath $TmpConfig -Force
    }
}
