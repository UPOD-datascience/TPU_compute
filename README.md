# TPU_compute

Basic scripts to
* spin up a TPU pod
* install prerequisites on the TPU VM
* execute a model script
  * the model script trains a Torch model on the TPU
  * using the data that is stored in the GCS bucket
  * the model script saves the model, including all checkpoints to the GCS bucket
  * during the training we have callback to Tensorboard

The folder structure is as follows:
```
TPU_compute
  README.md
  .env
    tpu\
      start_tpu.sh
      install_prerequisites.sh
      load_data.sh
    models\
      train_tokenizer.py
      train_model.py
```

Sources:
* https://cloud.google.com/tpu/docs/v5p-training#llama_2
