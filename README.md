# ML Engineering: DistilBERT Fine-Tuning

## Overview
This repository contains my work for the "Machine Learning Engineering: From NLP to Production" course, designed to transition from a Data Scientist with NLP/LLM expertise to an ML Engineer. The course focuses on building, deploying, optimizing, and scaling NLP systems for production.

## Module 0: Essential CS Tools (The Missing Semester)
- **Status**: In progress (Week 1: April 21, 2025 – April 27, 2025).
- **Objective**: Learn practical CS tools (shell scripting, Git, data wrangling, debugging) to enhance productivity for ML workflows.
- **Tasks**:
  - Write a shell script to automate Sentiment140 preprocessing.
  - Use Git for branching and merging.
  - Filter a JSON dataset with `jq`.
  - Debug a Python script.
- **Deliverables**: Shell script, filtered JSON, debugged script, Git branch history (to be updated upon completion).

## Module 1: Foundations of ML Engineering
- **Status**: Completed (April 21, 2025).
- **Objective**: Fine-tune DistilBERT for sentiment analysis on the Sentiment140 dataset, achieving >80% accuracy.
- **Dataset**: Sentiment140 (10,000 samples used). [Sentiment140 on Kaggle](https://www.kaggle.com/kazanova/sentiment140)
- **Results**:
  - **Final Eval Accuracy**: 0.8055 (80.55%, meets >80% requirement).
  - **Training Metrics**:
    | Epoch | Training Loss | Validation Loss | Accuracy  |
    |-------|---------------|-----------------|-----------|
    | 1     | 0.4276        | 0.4234          | 0.8055    |
    | 2     | 0.3269        | 0.4661          | 0.7900    |
  - Early stopping triggered after epoch 2 (accuracy dropped from 0.8055 to 0.7900).
  - Training time: ~48 minutes (CPU), 1000 steps, ~178 trillion FLOPs.
- **Steps Taken**:
  - Adjusted `TrainingArguments`: `num_train_epochs=3`, `weight_decay=0.01`, `learning_rate=2e-5`, `metric_for_best_model="eval_accuracy"`.
  - Added `EarlyStoppingCallback(early_stopping_patience=1)` to prevent overfitting.
- **Challenges Faced**:
  - `numpy` compilation issues: Resolved by installing `numpy<1.27` and Visual Studio Build Tools (`cl.exe`).
  - `labels` error: Fixed by ensuring correct label mapping in the dataset.
  - `evaluation_strategy` error: Set to `"epoch"` with matching `save_strategy`.
  - Overfitting: Mitigated with early stopping, weight decay, and lower learning rate (validation loss still increased slightly from 0.4234 to 0.4661).
- **Deliverables**:
  - Jupyter notebook: `notebooks/twitter_sentiments.ipynb`.
  - Saved model: `C:/Users/Alex Chung/Documents/ml_engineering_clean/final_model`.
- **Next Steps**:
  - Optionally scale to the full Sentiment140 dataset (1.6M samples) with distributed training in Module 5 (using AWS EC2 with GPUs).
  - Proceed to Module 2: Deploy the model as a FastAPI endpoint on AWS EC2.

## Future Modules
- **Module 2: Model Deployment and APIs** (April 28 – May 4, 2025): Deploy the fine-tuned model as a FastAPI endpoint on AWS EC2.
- **Module 3: Optimization, Scaling, and Monitoring** (May 5 – May 18, 2025): Optimize the model (e.g., ONNX), add monitoring, and perform A/B testing. **Planned Improvements**: Hyperparameter tuning with `optuna`, increase dropout to further reduce overfitting.
- **Module 4: MLOps and CI/CD** (May 12 – May 25, 2025): Set up CI/CD with GitHub Actions, MLflow, and DVC.
- **Module 5: System Design and Infrastructure** (May 19 – June 1, 2025): Design an end-to-end pipeline with AWS EC2 and SageMaker. **Planned Improvements**: GPU acceleration, train on the full Sentiment140 dataset.
- **Capstone: End-to-End NLP Application** (June 2 – June 8, 2025): Build and deploy a news summarizer with CI/CD and monitoring.

## Setup
- **Environment**: Python 3.10, `transformers` 4.35.2, `numpy` 1.26.4.
- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
