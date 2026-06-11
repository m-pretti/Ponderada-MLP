import numpy as np
from .losses import cross_entropy_loss

def train(model, x_train, y_train, epochs=5, batch_size=64):
    for epoch in range(epochs):
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        epoch_loss = 0
        for i in range(0, len(x_train), batch_size):
            x_b, y_b = x_train[i:i+batch_size], y_train[i:i+batch_size]
            out = model.forward(x_b)
            epoch_loss += cross_entropy_loss(out, y_b)
            grad = (out - y_b) / len(x_b)
            model.backward(grad)
        print(f"Epoch {epoch+1} Loss: {epoch_loss/(len(x_train)/batch_size):.4f}")

def train_with_history(model, x_train, y_train, epochs=5, batch_size=64):
    history = []
    for epoch in range(epochs):
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        epoch_loss = 0
        for i in range(0, len(x_train), batch_size):
            x_b, y_b = x_train[i:i+batch_size], y_train[i:i+batch_size]
            out = model.forward(x_b)
            loss = cross_entropy_loss(out, y_b)
            epoch_loss += loss
            grad = (out - y_b) / len(x_b)
            model.backward(grad)
        avg_loss = epoch_loss / (len(x_train) / batch_size)
        history.append(avg_loss)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    return history