# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MaxText is a high-performance, scalable LLM implementation in Python/JAX targeting Google Cloud TPUs and GPUs. The codebase emphasizes simplicity while achieving high Model FLOPs Utilization (MFU) across distributed systems.

## Essential Commands

### Testing
```bash
# Run all unit tests
python3 -m pytest --pyargs MaxText.tests

# Run unit tests and linting together
bash unit_test_and_lint.sh

# Run specific test
python3 -m pytest MaxText/tests/specific_test.py
```

### Linting and Code Style
```bash
# Run linting
python3 -m pylint $(git ls-files '*.py')

# Format and check code style
bash code_style.sh

# Check style without modifying
bash code_style.sh --check
```

### Training
```bash
# Basic training command
python3 -m MaxText.train MaxText/configs/base.yml \
  run_name=experiment_name \
  model_name=llama2-7b \
  base_output_directory=gs://bucket/outputs
```

### Inference
```bash
# Run inference/decoding
python3 -m MaxText.decode MaxText/configs/base.yml \
  run_name=decode_test \
  checkpoint_path=gs://bucket/checkpoints
```

## Architecture

### Core Structure
- **MaxText/train.py**: Main training loop entry point
- **MaxText/decode.py**: Inference/decoding entry point
- **MaxText/layers/**: Model components (attention, embeddings, decoders, transformers)
- **MaxText/input_pipeline/**: Data loading and preprocessing
- **MaxText/inference/**: Inference utilities (KV cache, paged attention)
- **MaxText/configs/**: YAML configuration files
  - `base.yml`: Default configuration
  - `models/`: Model-specific configurations
- **MaxText/tests/**: Unit and integration tests

### Key Architectural Patterns
1. **Configuration-driven**: All model parameters and training settings are controlled via YAML configs
2. **Modular layers**: Each model component is implemented as a separate module in `layers/`
3. **Hardware abstraction**: Code abstracts TPU/GPU differences through JAX
4. **Distributed by design**: Training and inference scale across multiple devices using JAX's pmap/pjit

### Model Support
The codebase supports multiple model families through a unified interface:
- Llama family (2, 3, 3.1, 3.3, 4)
- Mistral/Mixtral (including MoE variants)
- Gemma (1, 2, 3)
- DeepSeek (v2, v3 with MoE)
- GPT-3
- Qwen3

Each model has specific configuration files in `MaxText/configs/models/`.

### Important Development Patterns
1. **Module invocation**: Always use `python3 -m MaxText.module` instead of direct file execution
2. **Configuration inheritance**: Model configs extend base.yml
3. **Checkpoint handling**: Uses Orbax for checkpoint management with GCS integration
4. **Quantization**: Supports int8, fp8, and mixed precision training/inference
5. **Attention variants**: Supports multiple attention implementations (dot_product, flash, splash)

### Testing Strategy
- Unit tests for individual components in `MaxText/tests/`
- End-to-end tests in `end_to_end/` directory for full training/inference pipelines
- Hardware-specific tests for TPU and GPU configurations
- Always run tests before submitting changes

### Common Development Tasks
1. **Adding a new model**: Create config in `configs/models/`, implement specific layers if needed
2. **Modifying training loop**: Edit `train.py` and related utilities in `MaxText/`
3. **Optimizing performance**: Focus on `layers/` implementations and attention mechanisms
4. **Debugging distributed training**: Use profiling tools and check TPU/GPU utilization metrics