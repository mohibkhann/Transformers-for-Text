# Transformers for Text Classification

## Overview
This repository focuses on **Transformer-based models** for text classification. The objective is to classify questions into six categories using **self-attention mechanisms and fine-tuned BERT models**. The project includes data preprocessing, model training, and performance evaluation.

## Dataset
The dataset used is a **question classification dataset**, where each question belongs to one of six coarse categories:

- Abbreviation (ABBR)
- Entity (ENTY)
- Description (DESC)
- Human (HUM)
- Location (LOC)
- Numeric (NUM)

A subset of **2,000** questions was used for training and validation:
- **80%** for training
- **10%** for validation
- **10%** for testing

## Project Structure
The implementation consists of Transformer-based models for text classification, including:
- **Multi-Head Self-Attention Mechanism**
- **Positional Encoding**
- **Fine-Tuning BERT with Prefix Prompt-Tuning**

## Methodology
### 1. Data Preprocessing
- Tokenized text data using **BERT Tokenizer**.
- Converted categorical labels into numerical format using **Label Encoding**.
- Split the dataset into training, validation, and test sets.
- Used **PyTorch's DataLoader** for batch processing.

### 2. Transformer-Based Model
- Implemented a **Transformer-based classifier**:
  - Multi-Head Self-Attention for capturing long-range dependencies.
  - Positional Encoding to retain sequence order.
  - Fully connected layers for classification.
- Used **Adam optimizer** with a learning rate of `1e-4`.
- Achieved significant performance improvement over traditional models.

### 3. Prefix Prompt-Tuning with Transformers
- **Fine-tuned a pre-trained BERT model** by adding prefix prompts.
- Designed a **PrefixTuningForClassification** class:
  - Added **learnable prefix embeddings** to the input sequence.
  - Concatenated prefix prompts to token embeddings.
  - Freezed BERT parameters, allowing only prefix prompts and classification layer to be trained.
- Achieved **95.95% accuracy** after 30 epochs.

## Results
| Model                     | Train Accuracy | Validation Accuracy |
|---------------------------|---------------|---------------------|
| Transformer Classifier    | ~97.3%        | ~95.9%              |
| Prefix Tuning (BERT)      | ~99.25%       | ~95.9%              |

## Requirements
Ensure the following dependencies are installed:

```bash
pip install torch transformers datasets numpy pandas scikit-learn matplotlib
