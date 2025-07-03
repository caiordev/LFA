import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ntm.ntm import NeuralTuringMachine
from automata.finite_automaton import FiniteAutomaton, NeuralFiniteAutomaton

def create_dfa_example():
    """
    Create a simple DFA that recognizes strings ending with '01'.
    
    Returns:
        FiniteAutomaton: A DFA that recognizes strings ending with '01'
    """
    states = {'q0', 'q1', 'q2'}
    alphabet = {'0', '1'}
    transitions = {
        ('q0', '0'): 'q1',
        ('q0', '1'): 'q0',
        ('q1', '0'): 'q1',
        ('q1', '1'): 'q2',
        ('q2', '0'): 'q1',
        ('q2', '1'): 'q0'
    }
    initial_state = 'q0'
    final_states = {'q2'}
    
    return FiniteAutomaton(states, alphabet, transitions, initial_state, final_states)

def generate_examples(dfa, num_positive=5000, num_negative=5000, max_length=12):
    """
    Generate examples for training and testing.
    
    Args:
        dfa (FiniteAutomaton): The DFA to generate examples for
        num_positive (int): Number of positive examples
        num_negative (int): Number of negative examples
        max_length (int): Maximum length of generated strings
        
    Returns:
        tuple: (positive_examples, negative_examples)
    """
    alphabet_list = list(dfa.alphabet)
    positive_examples = []
    negative_examples = []
    
    # Generate positive examples
    while len(positive_examples) < num_positive:
        length = np.random.randint(2, max_length + 1)  # At least length 2 to have a chance of ending with '01'
        example = ''.join(np.random.choice(alphabet_list) for _ in range(length))
        if dfa.process(example) and example not in positive_examples:
            positive_examples.append(example)
    
    # Generate negative examples
    while len(negative_examples) < num_negative:
        # bias toward shorter strings: geometric distribution
        length = np.random.geometric(p=0.4)
        length = int(min(length, max_length))
        example = ''.join(np.random.choice(alphabet_list) for _ in range(length))
        if not dfa.process(example) and example not in negative_examples:
            negative_examples.append(example)
    
    return positive_examples, negative_examples

def one_hot_encode_batch(examples, alphabet):
    """One-hot encode with explicit EOS '#' symbol."""
    if '#' not in alphabet:
        alphabet = alphabet + ['#']  # append EOS at end
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    max_len = max(len(ex) for ex in examples) + 1  # reserve slot for EOS
    batch_size = len(examples)
    result = torch.zeros(batch_size, max_len, len(alphabet))
    for i, ex in enumerate(examples):
        for j, ch in enumerate(ex):
            result[i, j, char_to_idx[ch]] = 1.0
        # EOS mark immediately after last real char
        result[i, len(ex), char_to_idx['#']] = 1.0
    return result
    """
    Convert a batch of examples to one-hot encoding.
    
    Args:
        examples (list): List of string examples
        alphabet (list): List of symbols in the alphabet
        
    Returns:
        torch.Tensor: One-hot encoded batch of shape (batch_size, max_len, alphabet_size)
    """
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    max_len = max(len(ex) for ex in examples)
    batch_size = len(examples)
    alphabet_size = len(alphabet)
    
    result = torch.zeros(batch_size, max_len, alphabet_size)
    for i, example in enumerate(examples):
        for j, char in enumerate(example):
            if char in char_to_idx:
                result[i, j, char_to_idx[char]] = 1.0
                
    return result

def train_neural_automaton(dfa, positive_examples, negative_examples, num_epochs=100):
    """
    Train a neural automaton to simulate a DFA using a Neural Turing Machine.
    
    Args:
        dfa (FiniteAutomaton): The DFA to simulate
        positive_examples (list): List of strings accepted by the DFA
        negative_examples (list): List of strings rejected by the DFA
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (neural_automaton, alphabet, losses)
    """
    # Create alphabet from examples
    alphabet = sorted(list(dfa.alphabet) + ['#'])
    
    # Parameters
    num_symbols = len(alphabet)
    
    # Create Neural Turing Machine-based DFA
    class NTMDFA(nn.Module):
        """Neural DFA implemented with a Neural Turing Machine core."""
        def __init__(self, input_size, controller_size=64, memory_size=32, memory_locations=64):
            super(NTMDFA, self).__init__()
            self.ntm = NeuralTuringMachine(
                input_size=input_size,
                output_size=1,
                controller_size=controller_size,
                memory_size=memory_size,
                memory_locations=memory_locations,
            )
        def forward(self, x_batch):
            # x_batch: (batch, seq_len, input_size) -> NTM expects (seq_len, batch, input_size)
            x_seq = x_batch.permute(1, 0, 2)
            outputs = self.ntm.forward_sequence(x_seq)
            return torch.sigmoid(outputs[-1])  # (batch, 1) probabilities
    neural_automaton = NTMDFA(input_size=num_symbols)
    
    # Prepare data
    all_examples = positive_examples + negative_examples
    labels = [1] * len(positive_examples) + [0] * len(negative_examples)
    
    # One-hot encode examples
    inputs = one_hot_encode_batch(all_examples, alphabet)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(inputs, labels_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    dataset_size = len(dataset)
    
    # Optimizer
    optimizer = optim.Adam(neural_automaton.parameters(), lr=0.001)  # Lower learning rate for NTM
    
    # Training loop (mini-batch)
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_inputs, batch_labels = batch
            optimizer.zero_grad()
            outputs = neural_automaton(batch_inputs)
            loss = nn.BCELoss()(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(neural_automaton.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item() * batch_labels.size(0)
        epoch_loss /= dataset_size
        losses.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    
    return neural_automaton, alphabet, losses

def test_neural_automaton(neural_automaton, alphabet, test_examples, expected_results):
    """
    Test the trained neural automaton on examples.
    
    Args:
        neural_automaton (NTMDFA): Trained neural automaton based on NTM
        alphabet (list): List of symbols in the alphabet
        test_examples (list): List of test strings
        expected_results (list): List of expected results (0 or 1)
        
    Returns:
        float: Accuracy of the model
    """
    # One-hot encode examples
    inputs = one_hot_encode_batch(test_examples, alphabet)
    
    # Forward pass
    with torch.no_grad():
        outputs = neural_automaton(inputs)
        predictions = (outputs > 0.6).float()
        
    # ------------------ Metrics ------------------
    targets = torch.tensor(expected_results).float()
    preds = predictions.squeeze()
    correct = (preds == targets).sum().item()
    accuracy = correct / len(test_examples)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # ------------------ Reporting ------------------
    for example, pred, expected in zip(test_examples, preds.tolist(), expected_results):
        print(f"Example: {example}, Prediction: {int(pred)}, Expected: {expected}")

    print("\nConfusion matrix (rows=Actual, cols=Pred):")
    print(f"            Pred 0   Pred 1")
    print(f"Actual 0   {tn:6d}   {fp:6d}")
    print(f"Actual 1   {fn:6d}   {tp:6d}\n")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 score : {f1:.2f}\n")

    return accuracy

def plot_training_loss(losses):
    """
    Plot the training loss.
    
    Args:
        losses (list): List of loss values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def main():
    """
    Main function to demonstrate automaton simulation using Neural Turing Machine.
    The NTM learns to recognize strings that end with '01' by developing its own
    internal memory representation, without explicit state transitions.
    """
    # Create a DFA
    print("Creating DFA that recognizes strings ending with '01'...")
    dfa = create_dfa_example()
    
    # Generate examples
    print("Generating examples...")
    positive_examples, negative_examples = generate_examples(dfa, num_positive=500, num_negative=500, max_length=12)
    
    # Print some examples
    print("\nPositive examples (accepted by DFA):")
    for i, example in enumerate(positive_examples[:5]):
        print(f"  {example}")
    print("  ...")
    
    print("\nNegative examples (rejected by DFA):")
    for i, example in enumerate(negative_examples[:5]):
        print(f"  {example}")
    print("  ...")
    
    # Train the neural automaton using NTM
    print("\nTraining Neural Turing Machine-based automaton...")
    neural_automaton, alphabet, losses = train_neural_automaton(
        dfa, positive_examples, negative_examples, num_epochs=100
    )
    
    # Plot training loss
    plot_training_loss(losses)
    print("Training loss plot saved as 'training_loss.png'")
    
    # Test the neural automaton
    print("\nTesting Neural Turing Machine-based automaton...")
    test_examples = [
        "01",      # Positive: ends with 01
        "101",     # Positive: ends with 01
        "0101",    # Positive: ends with 01
        "1",       # Negative: doesn't end with 01
        "10",      # Negative: doesn't end with 01
        "011",     # Negative: doesn't end with 01
        "0011",    # Negative: doesn't end with 01
        "00101",   # Positive: ends with 01
        "",        # Negative: empty string
        "0"        # Negative: doesn't end with 01
    ]
    expected_results = [1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  # 1 for accept, 0 for reject
    
    accuracy = test_neural_automaton(neural_automaton, alphabet, test_examples, expected_results)
    
    print(f"\nNeural Turing Machine DFA accuracy: {accuracy:.2f}")

    print("\nInteractive testing. Enter strings of 0s and 1s to test (empty line to quit).")
    while True:
        user_input = input("Input: ")
        if user_input == "":
            break
            
        # Validate input
        valid = True
        for ch in user_input:
            if ch not in ['0', '1']:
                print(f"Invalid character '{ch}'. Only '0' and '1' are allowed.")
                valid = False
                break
        if not valid:
            continue
            
        # Build one-hot encoded tensor for the entered string
        encoded = one_hot_encode_batch([user_input], alphabet)
        
        # Neural model prediction
        with torch.no_grad():
            output = neural_automaton(encoded)
            prob = output.item()
        neural_pred = prob > 0.6
        print(f"Neural TM probability: {prob:.2f} -> {'ACCEPT' if neural_pred else 'REJECT'}")
        
        # Classical DFA verification (for comparison)
        classical_pred = dfa.process(user_input)
        print(f"Classical DFA verdict: {'ACCEPT' if classical_pred else 'REJECT'}\n")
        print(f"Expected pattern: String should end with '01' to be accepted\n")

if __name__ == "__main__":
    main()
