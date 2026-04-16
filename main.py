import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torch

from src import dataloader, preprocess, eda, classical_models, deep_learning_models, utils
import src.config as config

def main():
    parser = argparse.ArgumentParser(description="IMDb Sentiment Analysis")
    parser.add_argument('--mode', type=str, required=True, 
                    choices=['eda', 'train_classical', 'test_classical', 'train_dl', 'test_dl', 'tune_classical'],
                    help="Operation mode")

    parser.add_argument('--model', type=str, default='lstm', 
                        choices=['nb', 'lr', 'svm', 'rnn', 'lstm'],
                        help="Specific model to train/test")
    args = parser.parse_args()

    classical_models_list = ['nb', 'lr', 'svm']
    dl_models_list = ['rnn', 'lstm']

    if args.mode in ['train_classical', 'test_classical'] and args.model not in classical_models_list:
        parser.error(f"Mode '{args.mode}' requires a classical model. Please choose from: {classical_models_list}")
    
    if args.mode in ['train_dl', 'test_dl'] and args.model not in dl_models_list:
        parser.error(f"Mode '{args.mode}' requires a deep learning model. Please choose from: {dl_models_list}")

    # Data Setup
    dataloader.extract_data(config.RAW_DATA_PATH)
    
    if 'train' in args.mode:
        print("Loading Training Data...")
        df = dataloader.load_to_dataframe('train')
        X_train, X_val, y_train, y_val = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)
    
    if 'test' in args.mode:
        print("Loading Test Data...")
        test_df = dataloader.load_to_dataframe('test')
        X_test, y_test = test_df['text'], test_df['sentiment']

    # --- CLASSICAL MODELS ---
    if args.mode == 'train_classical':
        if args.model == 'nb':
            model = classical_models.get_nb_pipeline(config.MAX_FEATURES_TFIDF)
        elif args.model == 'lr':
            model = classical_models.get_lr_pipeline(config.MAX_FEATURES_TFIDF)
        else:
            model = classical_models.get_svm_pipeline(config.MAX_FEATURES_TFIDF)
        
        print(f"Training {args.model}...")
        model.fit(X_train, y_train)
        utils.save_sklearn_model(model, f"{args.model}_model.pkl")
        print(f"Validation Accuracy: {model.score(X_val, y_val):.4f}")

    elif args.mode == 'test_classical':
        model = utils.load_sklearn_model(f"{args.model}_model.pkl")
        if model:
            y_pred = model.predict(X_test)
            print(f"Test Accuracy for {args.model}: {accuracy_score(y_test, y_pred):.4f}")
            utils.plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix: {args.model.upper()}")


    # --- DEEP LEARNING ---
    elif args.mode == 'train_dl':
        vocab = preprocess.build_vocab(X_train)
        train_loader = DataLoader(preprocess.IMDBDataset(X_train, y_train, vocab), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(preprocess.IMDBDataset(X_val, y_val, vocab), batch_size=config.BATCH_SIZE)

        if args.model == 'rnn':
            model = deep_learning_models.SentimentRNN(config.MAX_VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM)
        else:
            model = deep_learning_models.SentimentLSTM(config.MAX_VOCAB_SIZE, config.EMBED_DIM, config.HIDDEN_DIM)

        history = utils.train_dl_model(model, train_loader, val_loader, config)
        utils.save_torch_model(model, vocab, f"{args.model}_model")
        utils.plot_training_history(history)

    elif args.mode == 'test_dl':
        # Determine class
        m_class = deep_learning_models.SentimentRNN if args.model == 'rnn' else deep_learning_models.SentimentLSTM
        model, vocab = utils.load_torch_model(m_class, config.MAX_VOCAB_SIZE, config.EMBED_DIM, 
                                             config.HIDDEN_DIM, f"{args.model}_model", config.DEVICE)
        
        test_loader = DataLoader(preprocess.IMDBDataset(X_test, y_test, vocab), batch_size=config.BATCH_SIZE)
        
        # Collect all predictions for confusion matrix
        all_preds = []
        all_true = []
        model.eval()
        with torch.no_grad():
            for texts, labels in test_loader:
                texts = texts.to(config.DEVICE)
                outputs = model(texts)
                all_preds.extend((outputs > 0.5).int().cpu().numpy())
                all_true.extend(labels.int().cpu().numpy())
        
        print(f"Test Accuracy for {args.model}: {accuracy_score(all_true, all_preds):.4f}")
        utils.plot_confusion_matrix(all_true, all_preds, title=f"Confusion Matrix: {args.model.upper()}")

    # ---HYPERPARAMETER TUNING---

    if args.mode == 'tune_classical':
        print(f"Starting Hyperparameter Tuning for {args.model.upper()}...")
        
        # Load and clean data (Tuning usually uses the full training set)
        df = dataloader.load_to_dataframe('train')
        X = df['text'].apply(preprocess.clean_text)
        y = df['sentiment']

        if args.model == 'nb':
            grid = classical_models.tune_nb(X, y)
        elif args.model == 'lr':
            grid = classical_models.tune_lr(X, y)
        elif args.model == 'svm':
            grid = classical_models.tune_svm(X, y)
        else:
            print("Please select a classical model: nb, lr, or svm")
            return

        print("\n" + "="*30)
        print(f"Best Score: {grid.best_score_:.4f}")
        print(f"Best Params: {grid.best_params_}")
        print("="*30)

        # Save the best model found
        best_model = grid.best_estimator_
        utils.save_sklearn_model(best_model, f"{args.model}_model.pkl")
        print(f"Best {args.model.upper()} model saved to models/{args.model}_model.pkl")


    elif args.mode == 'eda':
        print("Generating Exploratory Analysis Plots...")
        df = dataloader.load_to_dataframe('train')
        
        # 1. Distribution Plots (Word counts / Avg length)
        eda.plot_distributions(df)
        
        # 2. Vectorize for advanced visuals
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        # We use cleaned text to ensure the clouds aren't full of <br /> tags
        cleaned_docs = df['text'].apply(preprocess.clean_text)
        matrix = tfidf.fit_transform(cleaned_docs)
        features = tfidf.get_feature_names_out()
        
        # 3. LSA 2D Projection
        eda.plot_lsa(matrix, df['sentiment'])
        
        # 4. Polarity Bar Chart
        eda.plot_word_importance(matrix, features, df['sentiment'])
        
        # 5. Word Clouds (NEW)
        print("Generating Word Clouds...")
        eda.plot_wordclouds(matrix, features, df['sentiment'])

if __name__ == "__main__":
    main()