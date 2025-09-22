# Load environment variables from ../.env
Get-Content ../.env | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

# Verify that the working directory exists
if (-not (Test-Path $env:FULL_SCRIPT_DIR)) {
    Write-Error "Mapped drive $env:FULL_SCRIPT_DIR does not exist."
    popd
    exit
}

Write-Output "Starting training..."

$cmd = 'cd /d ' + $env:FULL_SCRIPT_DIR + ' && poetry run python ../models/cpt_llama_gpu.py ' +
       '--dataset_dir=' + $env:DATASET_FOLDER + ' ' +
       '--tmp_dir=' + $env:TMP_DIR + ' ' +
       '--output_dir=' + $env:OUTPUT_DIR + ' ' +
       '--model_name=' + $env:MODEL_NAME + ' ' +
       '--tokenizer_name_or_path=' + $env:TOKENIZER_PATH + ' ' +
	   '--dataset_format=json' + ' ' + 
	   '--wandb_key=' + $env:WANDB_KEY + ' ' + 
       '--per_device_train_batch_size=1 ' +
       '--gradient_accumulation_steps=64 ' +
       '--save_epoch_percentage=0.005 ' +
       '--logging_steps=500 ' +
       '--num_warmup_steps=50000 ' +
       '--max_seq_length=' + $env:MAX_SEQ_LEN + ' ' +
       '--learning_rate=1e-4 ' +
       '--keep_in_memory ' +
       '--weight_decay=0.0000001 ' +
       '--num_train_epochs=1 ' +
	   '--bf16'

Start-Process -NoNewWindow -FilePath "cmd.exe" -ArgumentList "/k", $cmd -Wait
Write-Output "Training started."