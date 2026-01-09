# DeepSeek V3-671B Test Script Arguments Documentation

This document provides detailed documentation of all arguments used in the DeepSeek V3-671B test script for TPU v5p-256.

## Environment Variables
- `MODEL_NAME='deepseek3-671b'`: Model identifier used throughout the script
- `TOKENIZER_PATH='deepseek-ai/DeepSeek-V3'`: HuggingFace path to the tokenizer
- `CHKPT_BUCKET`: GCS bucket containing HuggingFace checkpoint files
- `MODEL_BUCKET`: GCS bucket for MaxText-compatible model weights
- `DATASET_PATH`: GCS bucket containing training data
- `BASE_OUTPUT_DIRECTORY`: GCS bucket for MaxText output files (logs, checkpoints, etc.)
- `CONVERTED_CHECKPOINT`: Path to scanned checkpoint (for training)
- `UNSCANNED_CKPT_PATH`: Path to unscanned checkpoint (for efficient decoding)

## Command 1: Convert HuggingFace Checkpoint to MaxText Format

```bash
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_ckpt \
  --base_model_path ${CHKPT_BUCKET} \
  --maxtext_model_path ${MODEL_BUCKET}/${idx} \
  --model_size ${MODEL_NAME}
```

### Arguments:
- **`JAX_PLATFORMS=cpu`**: Forces JAX to run on CPU only (no GPU/TPU needed for conversion)
- **`--base_model_path`**: Source path to HuggingFace checkpoint files
- **`--maxtext_model_path`**: Destination path for converted MaxText checkpoint
- **`--model_size`**: Model variant identifier (deepseek3-671b)

### Purpose:
Converts the original DeepSeek checkpoint from HuggingFace format to MaxText's "scanned" format, which is optimized for training but not for inference.

## Command 2: Convert to Unscanned Checkpoint Format

```bash
JAX_PLATFORMS=cpu python3 -m MaxText.convert_deepseek_unscanned_ckpt \
  --base_model_path ${CHKPT_BUCKET} \
  --maxtext_model_path ${MODEL_BUCKET}/${idx}/unscanned \
  --model_size ${MODEL_NAME}
```

### Arguments:
- **`JAX_PLATFORMS=cpu`**: CPU-only execution for conversion
- **`--base_model_path`**: Source HuggingFace checkpoint path
- **`--maxtext_model_path`**: Destination for unscanned checkpoint (note `/unscanned` suffix)
- **`--model_size`**: Model variant (deepseek3-671b)

### Purpose:
Creates an "unscanned" version of the checkpoint optimized for inference performance. Unscanned format is more efficient for decoding as it doesn't require JAX's scan transformations.

## Command 3: Pre-training with Matmul Implementation

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  run_name=matmul_pre_training \
  per_device_batch_size=4 \
  enable_checkpointing=false \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=128 \
  steps=5 \
  max_target_length=1024 \
  async_checkpointing=false \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  attention=flash \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  megablox=False \
  sparse_matmul=False \
  dataset_type=synthetic
```

### Arguments:
- **`MaxText/configs/base.yml`**: Base configuration file
- **`base_output_directory`**: GCS path for outputs (logs, checkpoints)
- **`run_name=matmul_pre_training`**: Unique identifier for this training run
- **`per_device_batch_size=4`**: Batch size per TPU chip (4 examples)
- **`enable_checkpointing=false`**: Disables saving checkpoints during training
- **`model_name`**: DeepSeek model variant (deepseek3-671b)
- **`ici_fsdp_parallelism=128`**: Fully Sharded Data Parallelism across 128 devices
- **`steps=5`**: Only 5 training steps (for testing)
- **`max_target_length=1024`**: Maximum sequence length for targets
- **`async_checkpointing=false`**: Synchronous checkpointing (when enabled)
- **`tokenizer_type=huggingface`**: Use HuggingFace tokenizer
- **`tokenizer_path`**: Path to tokenizer on HuggingFace
- **`attention=flash`**: Use FlashAttention implementation
- **`dtype=bfloat16`**: Activation data type
- **`weight_dtype=bfloat16`**: Model weight data type
- **`megablox=False`**: Disable Megablox (optimized MoE implementation)
- **`sparse_matmul=False`**: Use dense matrix multiplication
- **`dataset_type=synthetic`**: Use synthetic data for testing

## Command 4: Fine-tuning with Matmul Implementation

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  dataset_path=${DATASET_PATH} \
  load_parameters_path=${CONVERTED_CHECKPOINT} \
  run_name=matmul_fine_tuning \
  per_device_batch_size=4 \
  enable_checkpointing=false \
  model_name=${MODEL_NAME} \
  ici_fsdp_parallelism=128 \
  steps=5 \
  max_target_length=1024 \
  async_checkpointing=false \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  attention=flash \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  megablox=False \
  sparse_matmul=False \
  enable_checkpointing=true
```

### Additional/Changed Arguments:
- **`dataset_path`**: Real dataset path (not synthetic)
- **`load_parameters_path`**: Pre-trained checkpoint to start from
- **`run_name=matmul_fine_tuning`**: Different run identifier
- **`enable_checkpointing=true`**: Enables checkpoint saving (note: appears twice with different values)

## Command 5: Supervised Fine-tuning (SFT)

```bash
python3 -m MaxText.sft_trainer MaxText/configs/sft.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=${CONVERTED_CHECKPOINT} \
  run_name=matmul_supervised_fine_tuning \
  per_device_batch_size=4 \
  enable_checkpointing=false \
  model_name=${MODEL_NAME} \
  steps=5 \
  max_target_length=1024 \
  async_checkpointing=false \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  attention=flash \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  megablox=False \
  sparse_matmul=False \
  enable_checkpointing=true \
  ici_expert_parallelism=128 \
  ici_fsdp_parallelism=1 \
  dataset_type=hf
```

### Key Differences:
- **`MaxText/configs/sft.yml`**: Uses SFT-specific config
- **`MaxText.sft_trainer`**: Different training module for supervised fine-tuning
- **`ici_expert_parallelism=128`**: Expert parallelism for MoE layers (128-way)
- **`ici_fsdp_parallelism=1`**: No FSDP (since using expert parallelism)
- **`dataset_type=hf`**: HuggingFace dataset format

## Command 6: Decoding (Inference)

```bash
python3 -m MaxText.decode MaxText/configs/base.yml \
  base_output_directory=${BASE_OUTPUT_DIRECTORY} \
  load_parameters_path=${UNSCANNED_CKPT_PATH} \
  run_name=decode \
  per_device_batch_size=1 \
  enable_checkpointing=false \
  model_name=${MODEL_NAME} \
  max_prefill_predict_length=100 \
  max_target_length=1024 \
  tokenizer_type=huggingface \
  tokenizer_path=${TOKENIZER_PATH} \
  attention=flash \
  dtype=bfloat16 \
  weight_dtype=bfloat16 \
  megablox=False \
  sparse_matmul=False \
  ici_tensor_parallelism=128 \
  ici_fsdp_parallelism=1 \
  prompt="I love to" \
  scan_layers=False
```

### Decoding-Specific Arguments:
- **`load_parameters_path=${UNSCANNED_CKPT_PATH}`**: Uses unscanned checkpoint for efficiency
- **`per_device_batch_size=1`**: Single example per device for decoding
- **`max_prefill_predict_length=100`**: Maximum tokens for prefill phase
- **`ici_tensor_parallelism=128`**: 128-way tensor parallelism for inference
- **`prompt="I love to"`**: Input prompt for generation
- **`scan_layers=False`**: Disable JAX scan transformations (using unscanned checkpoint)

## Key Observations

### Parallelism Strategy:
1. **Training**: Uses FSDP-128 (128-way Fully Sharded Data Parallelism)
2. **SFT**: Uses Expert Parallelism-128 (for MoE layers)
3. **Inference**: Uses Tensor Parallelism-128 (optimal for decoding)

### Model Configuration:
- DeepSeek V3 is a 671B parameter Mixture of Experts (MoE) model
- Uses FlashAttention for all operations
- BFloat16 precision throughout
- 1024 token context length for testing

### Testing Strategy:
- Only 5 steps for each phase (minimal testing)
- Synthetic data for pre-training test
- Real data for fine-tuning and SFT
- Tests the full pipeline: conversion → pre-training → fine-tuning → SFT → inference

### Performance Considerations:
- Scanned checkpoints for training (better for JAX transformations)
- Unscanned checkpoints for inference (better performance)
- Different parallelism strategies optimized for each phase
- CPU-only checkpoint conversion to avoid TPU allocation