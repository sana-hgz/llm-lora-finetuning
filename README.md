# llm-lora-finetuning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Parameter-efficient fine-tuning of Qwen2.5-0.5B using LoRA and 4-bit quantization for French text summarization on the OrangeSum dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Results](#results)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project demonstrates efficient fine-tuning of a 500M parameter language model (Qwen2.5-0.5B) for French summarization using:
- **LoRA (Low-Rank Adaptation)**: Reducing trainable parameters by 87.8%
- **4-bit Quantization**: Cutting memory usage by 49%
- **Completion-Only Training**: Focusing learning on response generation

**Key Achievement**: Successfully fine-tuned a 500M parameter model on a single T4 GPU (15GB VRAM) with only 26.5M trainable parameters.

## ✨ Features

- 🚀 **Memory-Efficient**: Uses 4-bit quantization (NF4) and LoRA
- 📊 **Complete Pipeline**: Data preprocessing → Training → Evaluation → Model merging
- 🎓 **Educational**: Well-documented notebook with theoretical explanations
- 🌍 **Multilingual Ready**: Easily adaptable to other languages
- 💾 **Optimized**: Gradient accumulation, mixed-precision training (BF16)


## 📊 Results

### Memory Optimization

| Configuration | Parameters | VRAM Usage | Reduction |
|--------------|------------|------------|-----------|
| Full Fine-tuning | 216M (43.2%) | 2.28 GB | - |
| LoRA + Quantization | 26.5M (5.3%) | 1.16 GB | **49%** |

### Model Performance

- **Dataset**: OrangeSum (5,000 French article-summary pairs)
- **Training Steps**: 200
- **Inference Time**: ~18s for 200 tokens on T4 GPU
- **Quality**: Coherent French summaries with proper structure

### Example Output

**Input Article** (truncated):
> La France doit transposer une directive européenne sur la transparence salariale...

**Generated Summary**:
> La France doit transmettre une directive européenne sur la transparence salariale, qui vise à réduire les inégalités entre les femmes et les hommes.

## 🔧 Technical Details

### Architecture
- **Base Model**: Qwen2.5-0.5B (500M parameters)
- **LoRA Configuration**: 
  - Rank (r): 32
  - Alpha: 64
  - Dropout: 0.05
  - Target Modules: All attention and MLP layers

### Training Configuration
- **Quantization**: 4-bit NF4 with double quantization
- **Compute dtype**: BFloat16
- **Optimizer**: Paged AdamW 8-bit
- **Learning Rate**: 5e-4 with cosine scheduler
- **Batch Size**: 1 per device (effective: 16 with gradient accumulation)
- **Epochs**: 2 (200 steps)

### Key Optimizations
1. **Completion-Only Training**: Labels masked with `-100` for prompt tokens
2. **Dynamic Padding**: Using `DataCollatorForSeq2Seq`
3. **Mixed Precision**: BF16 for faster computation
4. **Gradient Accumulation**: Simulating larger batch sizes


## 🎓 Learning Outcomes

This project covers:
- ✅ Parameter-Efficient Fine-Tuning (PEFT) techniques
- ✅ Model quantization and memory optimization
- ✅ Handling multilingual NLP tasks
- ✅ GPU resource management
- ✅ Chat template implementation
- ✅ Completion-only training strategies

## 🤝 Acknowledgments

- **Course**: APM 53674: ALTeGraD
- **Instructors**: Prof. Michalis Vazirgiannis, Dr. Hadi Abdine, Yang Zhang
- **Institution**: École Polytechnique / Telecom Paris
- **Dataset**: [OrangeSum](https://huggingface.co/datasets/giuliadc/orangesum_5k) by giuliadc
- **Base Model**: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) by Alibaba Cloud


## 🔗 Links

- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

## 📧 Contact

**Sana Hagaza**
- Email: hagazasana@gmail.com

---

⭐ If you find this project helpful, please consider giving it a star!

**Keywords**: LLM, Fine-tuning, LoRA, PEFT, Quantization, NLP, French, Summarization, PyTorch, HuggingFace, Transformers
