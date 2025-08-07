def update_trust(prior_trust, accuracy, decay=0.9):
    """Update trust with decay for declining accuracy."""
    likelihood = accuracy
    if accuracy < prior_trust:  # Decay if accuracy drops
        likelihood *= decay
    trust = (prior_trust * likelihood) / (prior_trust * likelihood + (1 - prior_trust) * (1 - likelihood))
    return max(0.5, min(trust, 0.99))  # Cap at 0.99 to avoid 1.0
