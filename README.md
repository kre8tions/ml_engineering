# ML Engineering: DistilBERT Fine-Tuning 
## Overview 
Fine-tuning DistilBERT on Sentiment140 dataset for binary sentiment classification. 
## Current Results 
- Peak validation accuracy: 0.8075 (epoch 2) 
- Final eval accuracy: 0.793 (epoch 1 model) 
- Overfitting observed (val loss 0.7392 at epoch 3) 
## Next Steps 
- Reduce epochs to 2, add weight decay, lower learning rate, optimize for accuracy. 
## Setup 
- Python 3.10, transformers 4.35.2, numpy 1.26.4 
- Install dependencies: `pip install -r requirements.txt` 
- Dataset: Download from [Kaggle: Sentiment140](https://www.kaggle.com/kazanova/sentiment140) 
