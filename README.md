# IMDb Movie Review Sentiment Classification

This project implements a complete NLP pipeline to classify IMDb movie reviews as positive or negative. It compares classical machine learning approaches (Naive Bayes, Logistic Regression, SVM) with Deep Learning architectures (RNN and LSTM).

## Project Structure
- `data/`: Placeholder for the IMDb dataset.
- `src/`: Source code modules for preprocessing, modeling, and evaluation.
- `models/`: Directory to save trained models.
- `main.py`: Entry point to run the pipeline.

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/SanidhyaKaushik/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Data Setup:**
   Download [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and place the aclImdb_v1.tar.gz file inside the data/ folder. The pipeline will handle extraction automatically.

## Usage
Run all the tasks via main.py using --mode and --model flags.
1. **Exploratory Data Analysis**
   Generate distribution plots, LSA 2D projections, and sentiment-polarized Word Clouds:
   ```bash
   python main.py --mode eda

2. **Hyperparameter Tuning**
   Find the best parameters for classical models (NB, LR, or SVM) using Grid Search:
   ```bash
   python main.py --mode tune_classical --model svm

3. **Classical Machine Leaarning**
   Train or test traditional models using TF-IDF features:
   ```bash
   # Training
   python main.py --mode train_classical --model lr
   # Testing (includes Confusion Matrix)
   python main.py --mode test_classical --model lr

4. **Deep Learning (PyTorch)**
   Train or test RNN/LSTM models. These models use an Embedding layer followed by Recurrent layers and Global Max Pooling:
   ```bash
   # Training
   python main.py --mode train_dl --model lstm

   # Testing (includes Confusion Matrix)
   python main.py --mode test_dl --model lstm