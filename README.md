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
    scripts\
      start_tpu.sh
      install_prerequisites.sh
      load_data.sh
    models\
      train_tokenizer.py
      train_bitnet.py
      train_deberta.py
      train_diffusion.py
      train_llama.py
      train_xlstm.py
```

Consider using [Tensorflow](https://huggingface.co/blog/tf_tpu) if Torch is not working for you.

Sources:
* https://cloud.google.com/tpu/docs/v5p-training#llama_2
* https://cloud.google.com/tpu/docs/pytorch-pods
* https://www.philschmid.de/getting-started-tpu-transformers#2-setup-jupyter-environment--install-transformers
* https://github.com/bramiozo/ModernBERT
* https://www.philschmid.de/fine-tune-modern-bert-in-2025
* https://discuss.huggingface.co/t/tpu-trainer-with-multi-core/16957
* https://www.kaggle.com/code/tanlikesmath/pytorch-tpu-starter-deberta-v3-large-training
* https://github.com/huggingface/transformers/issues/29659#issuecomment-2006524634
* https://pytorch.org/blog/quantization-aware-training/

# Pre-training scripts

[DeBERTav3 pre-training](https://github.com/microsoft/DeBERTa/tree/master/experiments/language_model)

[ModernBERT pre-training](https://github.com/AnswerDotAI/ModernBERT/tree/main)

[EuroBERT pre-training](https://github.com/Nicolas-BZRD/EuroBERT)

[Llama3 pre-training](https://github.com/vvr-rao/Training-a-Mini-114M-Parameter-Llama-3-like-Model-from-Scratch/blob/main/model%20and%20training%20code/train.py)

[QAT training](https://github.com/Qualcomm-AI-research/lr-qat)

[Llama3 finetuning](https://huggingface.co/blog/nroggendorff/train-with-llama-architecture)
