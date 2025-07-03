import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntm.ntm import NeuralTuringMachine
from formal_languages.regex_parser import NeuralRegexParser
import re
import random

def one_hot_encode(sequence, alphabet):
    """One-hot encode + add EOS '#' at end."""
    if '#' not in alphabet:
        alphabet = alphabet + ['#']
    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    seq = sequence + '#'
    result = torch.zeros(len(seq), len(alphabet))
    for i, ch in enumerate(seq):
        if ch in char_to_idx:
            result[i, char_to_idx[ch]] = 1.0
    return result
    """
    Convert a sequence to one-hot encoding.
    
    Args:
        sequence (str): Input sequence
        alphabet (list): List of symbols in the alphabet
        
    Returns:
        torch.Tensor: One-hot encoded sequence
    """
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    result = torch.zeros(len(sequence), len(alphabet))
    for i, char in enumerate(sequence):
        if char in char_to_idx:
            result[i, char_to_idx[char]] = 1.0
    return result

def train_regex_recognizer(regex_pattern, positive_examples, negative_examples, num_epochs=50):
    """
    Train a neural network to recognize strings matching a regex pattern.
    
    Args:
        regex_pattern (str): Regular expression pattern
        positive_examples (list): List of strings that match the pattern
        negative_examples (list): List of strings that don't match the pattern
        num_epochs (int): Number of training epochs
        
    Returns:
        NeuralRegexParser: Trained neural regex parser
    """
    # Create alphabet from examples
    alphabet = set()
    for example in positive_examples + negative_examples:
        alphabet.update(example)
    alphabet = sorted(alphabet)
    
    # Parameters
    input_size = len(alphabet)
    controller_size = 256
    memory_size = 32
    memory_locations = 128
    output_size = 1
    
    # Create Neural Turing Machine
    ntm = NeuralTuringMachine(
        input_size=input_size,
        output_size=controller_size,
        controller_size=controller_size,
        memory_size=memory_size,
        memory_locations=memory_locations,
        num_heads=1
    )
    
    # Create Neural Regex Parser
    parser = NeuralRegexParser(ntm, input_size, output_size)
    
    # Create optimizer
    optimizer = optim.Adam(parser.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Train the parser
    losses = parser.train_on_examples(positive_examples, negative_examples, optimizer, num_epochs)
    
    # Print final loss
    print(f"Final loss: {losses[-1]:.6f}")
    
    return parser, alphabet

def test_regex_recognizer(parser, alphabet, test_examples, expected_results):
    """
    Test the trained regex recognizer on examples.
    
    Args:
        parser (NeuralRegexParser): Trained neural regex parser
        alphabet (list): List of symbols in the alphabet
        test_examples (list): List of test strings
        expected_results (list): List of expected results (0 or 1)
        
    Returns:
        float: Accuracy of the model
    """
    correct = 0
    
    for example, expected in zip(test_examples, expected_results):
        # Convert example to one-hot encoding
        input_tensor = one_hot_encode(example, alphabet).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output = parser(input_tensor)
            prediction = (output > 0.5).float().item()
            
        # Check if prediction matches expected result
        if prediction == expected:
            correct += 1
            
        print(f"Example: {example}, Prediction: {prediction}, Expected: {expected}")
        
    accuracy = correct / len(test_examples)
    print(f"Accuracy: {accuracy:.2f}")
    
    return accuracy

def generate_dataset(regex_pattern, alphabet=('a','b'), max_len=12, n_pos=5000, n_neg=5000):
    pat = re.compile(regex_pattern + r'\Z')
    pos, neg = [], []
    while len(pos) < n_pos or len(neg) < n_neg:
        s = ''.join(random.choice(alphabet) for _ in range(random.randint(0, max_len)))
        if pat.fullmatch(s):
            if len(pos) < n_pos:
                pos.append(s)
        else:
            if len(neg) < n_neg:
                neg.append(s)
    return pos, neg

def main():
    """
    Main function to demonstrate regex recognition using Neural Turing Machine.
    """
    # Define regex pattern: strings with an even number of 'a's
    regex_pattern = "(b*ab*a)*b*"
    
    # Generate dataset automatically
    positive_examples, negative_examples = generate_dataset(regex_pattern)
    
    # Train the model
    print("Training regex recognizer...")
    parser, alphabet = train_regex_recognizer(regex_pattern, positive_examples, negative_examples, num_epochs=200)
    
    # Test the model
    print("\nTesting regex recognizer...")
    test_examples = ["", "a", "aa", "ab", "ba", "aabb", "abab", "baba", "bbbbbb", "ababab"]
    expected_results = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # 1 for match, 0 for no match
    
    accuracy = test_regex_recognizer(parser, alphabet, test_examples, expected_results)
    
    print(f"\nRegex pattern: {regex_pattern}")
    print(f"Model accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
