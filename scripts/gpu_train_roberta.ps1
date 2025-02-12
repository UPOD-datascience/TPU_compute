# Load environment variables from ../.env
Get-Content ../.env | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
}

Write-Output "Starting training..."

# Run the Python script in the background
Start-Process -NoNewWindow -FilePath "poetry" -ArgumentList @(
    "run", "python", "../models/cpt_roberta_gpu.py",
    "--dataset_dir=$env:DATASET_FOLDER",
    "--tmp_dir=$env:TMP_DIR",
    "--output_dir=$env:OUTPUT_DIR",
    "--model_name=$env:MODEL_NAME",
    "--tokenizer_name_or_path=$env:TOKENIZER_PATH",
    "--per_device_train_batch_size=16",
    "--gradient_accumulation_steps=20",
    "--save_epoch_percentage=0.5",
    "--logging_steps=5",
    "--num_warmup_steps=2000",
    "--num_cores=8",
    "--max_seq_length=$env:MAX_SEQ_LEN",
    "--learning_rate=0.0001",
    "--keep_in_memory",
    "--sharded_data",
    "--shuffle_buffer_size=10000",
    "--weight_decay=0.001",
    "--wandb_key=$env:WANDB_KEY",
    "--num_train_epochs=5"
) -PassThru | Out-Null
