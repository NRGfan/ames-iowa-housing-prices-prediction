# ames-iowa-housing-prices-prediction
This is a Kaggle competition, this repo consists of my attempts at predicting housing prices accurately with various regression techniques 

1. Baseline_random_forest.py: I implemented random forest using sklearn library, which achieved a score on kaggle of 0.15515 after some tuning
2. MLP regressor training.py: I implemented a two layer Multi Layer Perceptron (MLP) trained for 30 epochs using pytorch which achieved poor performance on kaggle at 0.639 

Table of results:

| Model | Remarks | Performance (kaggle score) |
|-------|---------|----------------------------|
| random forest| some more tuning needed | 0.15515 |
| MLP | training more epochs | 0.639 |
