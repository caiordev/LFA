import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from automata.pushdown_automaton import PushdownAutomaton, NeuralPushdownAutomaton
from ntm.ntm import NeuralTuringMachine

def create_balanced_parentheses_pda():
    """
    Create a pushdown automaton that recognizes balanced parentheses.
    """
    states = {'q0', 'q1', 'qaccept', 'qreject'}
    input_alphabet = {'(', ')'}
    stack_alphabet = {'$', '('}  # $ is the bottom of stack marker
    transitions = {
        ('q0', '(', '$'): [('q0', ['(', '$'])],  # Push '(' onto stack
        ('q0', '(', '('): [('q0', ['(', '('])],  # Push '(' onto stack
        ('q0', ')', '('): [('q0', [])],          # Pop '(' from stack
        ('q0', '', '$'): [('qaccept', ['$'])],   # Empty string is accepted
        ('q0', '', '('): [('qreject', ['('])],   # Unmatched '(' at the end
        ('q0', ')', '$'): [('qreject', ['$'])],  # Unmatched ')' at the beginning
    }
    initial_state = 'q0'
    initial_stack = ['$']
    final_states = {'qaccept'}
    
    return PushdownAutomaton(states, input_alphabet, stack_alphabet, transitions, initial_state, initial_stack, final_states)

def test_pushdown_automaton():
    """
    Test the pushdown automaton implementation.
    """
    print("Creating a pushdown automaton that recognizes balanced parentheses...")
    pda = create_balanced_parentheses_pda()
    
    # Test strings
    test_strings = ["", "()", "(())", "()()", "((()))", "(()())", ")", "(", "())", "(()"]
    
    print("\nTesting pushdown automaton on examples:")
    for s in test_strings:
        accepted = pda.process(s)
        print(f"String: '{s}', Accepted: {accepted}")
    
    print("\nTraining neural pushdown automaton...")
    
    print("\nSimulating neural pushdown automaton training...")
    
    # Create a simplified version for testing
    class NTMPDA(nn.Module):
        """Neural Pushdown Automaton implemented with a Neural Turing Machine core."""
        def __init__(self, input_size, controller_size=64, memory_size=32, memory_locations=64):
            super(NTMPDA, self).__init__()
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
            return outputs[-1]  # (batch, 1) logits
    
    # Prepare training data
    alphabet = sorted(list(pda.input_alphabet) + [''])  # Add epsilon
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    input_size = len(char_to_idx)
    model = NTMPDA(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ------------------ Generate dataset ------------------
    def _generate_balanced(n_pairs: int) -> str:
        """Create a random balanced-parentheses string with n_pairs pairs."""
        seq = []
        open_needed = n_pairs
        stack = 0
        for _ in range(2 * n_pairs):
            # Always push if no opens left on stack
            if stack == 0:
                seq.append('(')
                stack += 1
                open_needed -= 1
                continue
            # Decide randomly whether to push or pop
            if open_needed > 0 and np.random.rand() < 0.5:
                seq.append('(')
                stack += 1
                open_needed -= 1
            else:
                seq.append(')')
                stack -= 1
        # close remaining
        seq.extend(')' * stack)
        return ''.join(seq)

    def generate_dataset(pda, num_samples, max_pairs=6):
        """Generate a balanced dataset (exactly 50% +ve / -ve) quickly."""
        positives, negatives = [], []
        # positives using constructive generator
        while len(positives) < num_samples // 2:
            pairs = np.random.randint(0, max_pairs + 1)
            s = _generate_balanced(pairs)
            positives.append(s)
        # negatives by random sampling until rejected
        alphabet_chars = ['(', ')']
        while len(negatives) < num_samples // 2:
            length = np.random.randint(0, 2 * max_pairs + 1)
            s = ''.join(np.random.choice(alphabet_chars, size=length))
            if not pda.process(s):
                negatives.append(s)
        samples = positives + negatives
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        return samples, labels
    
    train_strings, train_labels = generate_dataset(pda, 1000, max_pairs=6)
    test_strings, test_labels = generate_dataset(pda, 200, max_pairs=6)
    
    # Determine maximum sequence length for padding
    max_len = max(max(len(s) for s in train_strings), max(len(s) for s in test_strings), 1)
    
    # One-hot encoding helpers
    def encode_batch(strings):
        batch = torch.zeros(len(strings), max_len, input_size)
        for i, s in enumerate(strings):
            if len(s) == 0:
                batch[i, 0, char_to_idx['']] = 1.0  # PAD symbol
            else:
                for j, ch in enumerate(s):
                    batch[i, j, char_to_idx[ch]] = 1.0
        return batch
    
    train_inputs = encode_batch(train_strings)
    test_inputs = encode_batch(test_strings)
    
    train_targets = torch.tensor(train_labels).unsqueeze(1)
    test_targets = torch.tensor(test_labels).unsqueeze(1)

    # ---- Balanced loss ----
    pos_fraction = train_targets.mean().item()
    neg_fraction = 1 - pos_fraction
    pos_weight = torch.tensor(neg_fraction / (pos_fraction + 1e-8))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    num_epochs = 100
    losses = []
    
    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            # Forward
            outputs = model(train_inputs)
            loss = criterion(outputs, train_targets)
            # Backward
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        print(f"Final training loss: {losses[-1]:.6f}")
        
        # Plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('pda_training_loss.png')
        print("Training loss plot saved as 'pda_training_loss.png'")
        
        print("\nTesting neural pushdown automaton on held-out data...")
        with torch.no_grad():
            outputs = model(test_inputs)
            probs = torch.sigmoid(outputs)
            predictions = (probs > 0.5).float()

            tp = ((predictions == 1) & (test_targets == 1)).sum().item()
            tn = ((predictions == 0) & (test_targets == 0)).sum().item()
            fp = ((predictions == 1) & (test_targets == 0)).sum().item()
            fn = ((predictions == 0) & (test_targets == 1)).sum().item()

            accuracy  = (tp + tn) / len(test_strings)
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)

            print("Confusion matrix (rows=Actual, cols=Pred):")
            print(f"          Pred 0   Pred 1")
            print(f"Actual 0   {tn:6d}   {fp:6d}")
            print(f"Actual 1   {fn:6d}   {tp:6d}\n")
            print(f"Accuracy : {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall   : {recall:.2f}")
            print(f"F1 score : {f1:.2f}\n")

            # Show a few random examples
            for idx in np.random.choice(len(test_strings), size=20, replace=False):
                pred = int(predictions[idx].item())
                target = int(test_targets[idx].item())
                s = test_strings[idx]
                print(f"String: '{s}', Pred: {pred}, Expected: {target}")

            print("\nInteractive testing. Enter strings of parentheses to test (empty line to quit).")
            while True:
                user_input = input("Input: ")
                if user_input == "":
                    break

                # Build one-hot encoded tensor for the entered string (shape: 1 × max_len × input_size)
                encoded = torch.zeros(1, max_len, input_size)
                if len(user_input) == 0:
                    encoded[0, 0, char_to_idx['']] = 1.0
                else:
                    valid = True
                    for j, ch in enumerate(user_input[:max_len]):
                        if ch not in char_to_idx or ch == '':
                            print(f"Invalid character '{ch}'. Allowed characters are '(' and ')'.")
                            valid = False
                            break
                        encoded[0, j, char_to_idx[ch]] = 1.0
                    if not valid:
                        continue

                # Neural PDA prediction
                with torch.no_grad():
                    output = model(encoded)
                    prob = torch.sigmoid(output).item()
                neural_pred = prob > 0.5
                print(f"Neural PDA probability: {prob:.2f} -> {'ACCEPT' if neural_pred else 'REJECT'}")

                # Classical PDA verification (for comparison)
                classical_pred = pda.process(user_input)
                print(f"Classical PDA verdict: {'ACCEPT' if classical_pred else 'REJECT'}\n")
                
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    test_pushdown_automaton()
