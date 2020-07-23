import numpy as np

def model_replacement_attack(original_model, target_model, learning_rate, per_round):
    return per_round / learning_rate * (target_model - original_model)
