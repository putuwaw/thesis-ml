# thesis-ml

ML Development and Experiment for [Thesis](https://github.com/putuwaw/thesis).

## Features
- Dataset with annotation result from Penyuluh Bahasa Bali: [dataset](dataset/)
- Numpy-based implementation: [notebook.py](notebook.py)
  - TF-IDF
  - Multinomial Naive Bayes
  - Label Binarizer
  - Chi-square
  - SMOTE
  - split data, accuracy score, cross validation
  - Pipeline
- ML development and experiment: [model.ipynb](model.ipynb)

## Prerequisites
This project mainly using uv and Python, so you need:
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv)

## Installation
- Clone the repository:
```
git clone https://github.com/putuwaw/thesis-ml.git
```
- Install dependencies:
```
uv sync --dev
```
- Start building your own model!
