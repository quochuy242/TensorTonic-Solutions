import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Write code here
    if rng is None:
        rng = np.random.default_rng()

    random_val = rng.random()
    if random_val > epsilon:
        return np.argmax(q_values)
    else:
        return rng.choice(np.arange(len(q_values)))
        
