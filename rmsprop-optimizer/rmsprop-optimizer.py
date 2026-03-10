import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    # Convert inputs to NumPy arrays
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)

    # Step 1: Update running squared gradient average
    new_s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: Update parameters
    new_w = w - (lr * g) / (np.sqrt(new_s) + eps)

    return new_w, new_s