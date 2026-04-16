import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np



def plot_distributions(df):
    # Calculate metadata if not present
    if 'word_count' not in df.columns:
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    if 'avg_word_length' not in df.columns:
        df['avg_word_length'] = df['text'].apply(lambda x: len(x) / len(x.split()) if len(x.split()) > 0 else 0)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='word_count', hue='sentiment', kde=True, element="step", palette='viridis')
    plt.title('Distribution of Word Count')
    plt.xlim(0, 1000)

    plt.subplot(1, 2, 2)
    sns.kdeplot(data=df, x='avg_word_length', hue='sentiment', fill=True, palette='magma')
    plt.title('Distribution of Avg Word Length')
    plt.xlim(0, 10)
    plt.show()

def plot_lsa(tfidf_matrix, labels):
    svd = TruncatedSVD(n_components=2, random_state=42)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    lsa_df = pd.DataFrame(lsa_matrix, columns=['Component 1', 'Component 2'])
    lsa_df['sentiment'] = labels.values

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=lsa_df, x='Component 1', y='Component 2', hue='sentiment', alpha=0.3, palette={1: 'green', 0: 'red'})
    plt.title("LSA: 2D Projection of TF-IDF Vectors")
    plt.show()

def plot_word_importance(tfidf_matrix, features, labels):
    pos_mask = (labels == 1).values
    neg_mask = (labels == 0).values
    
    pos_mean = np.asarray(tfidf_matrix[pos_mask].mean(axis=0)).flatten()
    neg_mean = np.asarray(tfidf_matrix[neg_mask].mean(axis=0)).flatten()
    
    comp = pd.DataFrame({'word': features, 'diff': pos_mean - neg_mean})
    polarized = pd.concat([comp.sort_values(by='diff').head(15), comp.sort_values(by='diff').tail(15)])
    
    plt.figure(figsize=(12, 8))
    colors = ['red' if x < 0 else 'green' for x in polarized['diff']]
    plt.barh(polarized['word'], polarized['diff'], color=colors)
    plt.title("Word Polarity (Difference in Mean TF-IDF)")
    plt.xlabel("<- More Negative | More Positive ->")
    plt.show()

def plot_wordclouds(tfidf_matrix, features, labels):
    # Split the matrix by sentiment
    pos_mask = (labels == 1).values
    neg_mask = (labels == 0).values

    # Calculate average TF-IDF score for every word in each class
    pos_weights = np.asarray(tfidf_matrix[pos_mask].mean(axis=0)).flatten()
    neg_weights = np.asarray(tfidf_matrix[neg_mask].mean(axis=0)).flatten()

    # Create dictionaries of {word: score}
    pos_dict = dict(zip(features, pos_weights))
    neg_dict = dict(zip(features, neg_weights))

    # Setup the figure
    plt.figure(figsize=(20, 10))

    # Positive Word Cloud
    plt.subplot(1, 2, 1)
    wc_pos = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Greens', max_words=100).generate_from_frequencies(pos_dict)
    plt.imshow(wc_pos, interpolation='bilinear')
    plt.title("Positive Reviews Importance", fontsize=25)
    plt.axis('off')

    # Negative Word Cloud
    plt.subplot(1, 2, 2)
    wc_neg = WordCloud(width=800, height=400, background_color='white', 
                       colormap='Reds', max_words=100).generate_from_frequencies(neg_dict)
    plt.imshow(wc_neg, interpolation='bilinear')
    plt.title("Negative Reviews Importance", fontsize=25)
    plt.axis('off')

    plt.tight_layout()
    plt.show()