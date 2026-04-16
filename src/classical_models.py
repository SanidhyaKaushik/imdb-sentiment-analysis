from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

# --- Base Pipeline Creators ---

def get_nb_pipeline(max_features=5000, alpha=1.0):
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=max_features)),
        ('clf', MultinomialNB(alpha=alpha))
    ])

def get_lr_pipeline(max_features=5000, C=1.0, ngram_range=(1, 2)):
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)),
        ('clf', LogisticRegression(max_iter=1000, C=C))
    ])

def get_svm_pipeline(max_features=5000, C=1.0, ngram_range=(1, 2)):
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)),
        ('clf', LinearSVC(C=C, dual=False, max_iter=2000))
    ])

# --- Hyperparameter Tuning Functions ---

def tune_nb(X, y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])
    param_grid = {
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__alpha': [0.1, 0.5, 1.0]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    return grid

def tune_lr(X, y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])
    param_grid = {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 2)],
        'clf__C': [0.1, 1, 10],
        'clf__penalty': ['l1', 'l2']
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    return grid

def tune_svm(X, y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(dual=False, max_iter=2000))
    ])
    param_grid = {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.01, 0.1, 1, 10]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid.fit(X, y)
    return grid