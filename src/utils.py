import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def train_dl_model(model, train_loader, val_loader, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()
    model.to(config.DEVICE)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(config.EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(texts)

                v_loss = criterion(outputs, labels)
                running_val_loss += v_loss.item()

                correct += ((outputs > 0.5) == labels).sum().item()

        avg_val_loss = running_val_loss / len(val_loader)
        acc = correct / len(val_loader.dataset)

        # Save metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

    return history


def save_sklearn_model(model, filename):
    os.makedirs('models', exist_ok=True)
    path = os.path.join('models', filename)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_sklearn_model(filename):
    path = os.path.join('models', filename)
    if os.path.exists(path):
        return joblib.load(path)
    else:
        print(f"No model found at {path}")
        return None

def save_torch_model(model, vocab, filename):
    os.makedirs('models', exist_ok=True)
    # Save model state
    torch.save(model.state_dict(), os.path.join('models', filename + ".pt"))
    # Save vocab (essential for testing later)
    joblib.dump(vocab, os.path.join('models', filename + "_vocab.pkl"))
    print(f"Deep learning model and vocab saved.")

def load_torch_model(model_class, vocab_size, embed_dim, hidden_dim, filename, device):
    model = model_class(vocab_size, embed_dim, hidden_dim)
    model.load_state_dict(torch.load(os.path.join('models', filename + ".pt"), map_location=device))
    vocab = joblib.load(os.path.join('models', filename + "_vocab.pkl"))
    model.to(device)
    model.eval()
    return model, vocab

def evaluate_torch_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    
    ax2.plot(history['val_acc'], label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    plt.title(title)
    plt.show()