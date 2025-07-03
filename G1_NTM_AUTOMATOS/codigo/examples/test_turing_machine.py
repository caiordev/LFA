import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from automata.turing_machine import TuringMachine, NeuralTuringMachineSimulator
from ntm.ntm import NeuralTuringMachine

def create_binary_palindrome_tm():
    """
    Create a Turing machine that recognizes binary palindromes.
    A palindrome reads the same forwards and backwards, like '101', '11', '000'.
    """
    states = {'q0', 'q1', 'q2', 'q3', 'q4', 'qaccept', 'qreject'}
    tape_alphabet = {'0', '1', 'X', 'Y', '_'}  # '_' is the blank symbol
    transitions = {
        # Initial state: replace first symbol with X/Y and move right
        ('q0', '0'): ('q1', 'X', 'R'),
        ('q0', '1'): ('q1', 'Y', 'R'),
        ('q0', '_'): ('qaccept', '_', 'N'),  # Empty string is a palindrome
        ('q0', 'X'): ('q3', 'X', 'R'),  # Already processed
        ('q0', 'Y'): ('q3', 'Y', 'R'),  # Already processed
        
        # Move right until the end of the string
        ('q1', '0'): ('q1', '0', 'R'),
        ('q1', '1'): ('q1', '1', 'R'),
        ('q1', 'X'): ('q1', 'X', 'R'),
        ('q1', 'Y'): ('q1', 'Y', 'R'),
        ('q1', '_'): ('q2', '_', 'L'),  # Reached the end, move left
        
        # Check the last symbol and move left
        ('q2', '0'): ('qreject', '0', 'N'),  # Last symbol doesn't match first (X)
        ('q2', '1'): ('qreject', '1', 'N'),  # Last symbol doesn't match first (Y)
        ('q2', 'X'): ('qreject', 'X', 'N'),  # Mismatch
        ('q2', 'Y'): ('qreject', 'Y', 'N'),  # Mismatch
        ('q2', '0'): ('q2', '0', 'L'),  # Keep moving left
        ('q2', '1'): ('q2', '1', 'L'),  # Keep moving left
        ('q2', 'X'): ('q2', 'X', 'L'),  # Keep moving left
        ('q2', 'Y'): ('q2', 'Y', 'L'),  # Keep moving left
        
        # Found a match for the first symbol, go back to start
        ('q2', 'X'): ('q0', 'X', 'R'),  # Found matching 0
        ('q2', 'Y'): ('q0', 'Y', 'R'),  # Found matching 1
        
        # All symbols matched, check if we're done
        ('q3', 'X'): ('q3', 'X', 'R'),  # Skip already processed
        ('q3', 'Y'): ('q3', 'Y', 'R'),  # Skip already processed
        ('q3', '_'): ('qaccept', '_', 'N'),  # All matched, accept
        ('q3', '0'): ('qreject', '0', 'N'),  # Unprocessed symbol, reject
        ('q3', '1'): ('qreject', '1', 'N'),  # Unprocessed symbol, reject
    }
    initial_state = 'q0'
    blank_symbol = '_'
    final_states = {'qaccept'}
    
    return TuringMachine(states, tape_alphabet, transitions, initial_state, blank_symbol, final_states)

def test_turing_machine():
    """
    Test the Turing machine implementation.
    """
    print("Creating a Turing machine that recognizes binary palindromes...")
    tm = create_binary_palindrome_tm()
    
    # Test strings - including both palindromes and non-palindromes
    palindromes = ["", "0", "1", "00", "11", "000", "111", "010", "101", "0110"] 
    non_palindromes = ["01", "10", "001", "011", "100", "110", "0101", "1010", "0011"]
    test_strings = palindromes + non_palindromes
    
    print("\nTesting Turing machine on examples:")
    print("\nPalindromes (should be accepted):")
    for s in palindromes:
        accepted, final_tape, steps = tm.process(s)
        print(f"String: '{s}', Accepted: {accepted}, Steps: {steps}")
        print(f"  Final tape: {''.join(final_tape)}")
        
    print("\nNon-palindromes (should be rejected):")
    for s in non_palindromes:
        accepted, final_tape, steps = tm.process(s)
        print(f"String: '{s}', Accepted: {accepted}, Steps: {steps}")
        print(f"  Final tape: {''.join(final_tape)}")
    
    print("\nTraining neural Turing machine simulator...")
    
    # Create Neural Turing Machine
    input_size = 64  # O input real do NTM é o embedding concatenado de estado e símbolo
    controller_size = 64
    memory_size = 32
    memory_locations = 64
    num_states = len(tm.states)
    
    ntm = NeuralTuringMachine(
        input_size=input_size,
        output_size=controller_size,
        controller_size=controller_size,
        memory_size=memory_size,
        memory_locations=memory_locations,
        num_heads=1
    )
    
    # Create Neural Turing Machine Simulator
    neural_tm = NeuralTuringMachineSimulator(ntm, num_states, len(tm.tape_alphabet))
    
    # Create optimizer
    optimizer = optim.Adam(neural_tm.parameters(), lr=0.01)
    
    # Prepare training data
    alphabet = sorted(tm.tape_alphabet)
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    
    # Create one-hot encoded inputs and expected outputs
    inputs = []
    targets = []
    
    # Determine the maximum length of test strings
    max_len = max(len(s) for s in test_strings if len(s) > 0)
    
    # Store expected results for each string
    expected_results = {}
    for s in test_strings:
        accepted, _, _ = tm.process(s)
        expected_results[s] = accepted

    for s in test_strings:
        if len(s) == 0:
            # Skip empty examples to avoid concatenation errors
            continue
        # One-hot encode the input string, with padding
        input_tensor = torch.zeros(1, max_len, len(alphabet))
        for i, char in enumerate(s):
            input_tensor[0, i, char_to_idx.get(char, char_to_idx['_'])] = 1.0
        
        # Get expected output (1 for accept, 0 for reject)
        accepted = expected_results[s]
        target = torch.tensor([[1.0]]) if accepted else torch.tensor([[0.0]])
        
        inputs.append(input_tensor)
        targets.append(target)
    
    # Concatenate inputs and targets
    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)
    
    # Training loop
    num_epochs = 100
    losses = []
    
    try:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = neural_tm(inputs)
            # Make shape (batch, 1) to match targets
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            # Compute loss
            loss = nn.BCELoss()(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        print(f"Final loss: {losses[-1]:.6f}")
        
        # Test the neural Turing machine
        print("\nTesting neural Turing machine simulator...")
        with torch.no_grad():
            outputs, _ = neural_tm(inputs)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            predictions = (outputs > 0.5).float()
            
            # Create a list of test strings that were actually used (excluding empty string)
            used_test_strings = [s for s in test_strings if len(s) > 0]
            
            print("\nPalindrome detection results:")
            print("-" * 50)
            print(f"{'String':^10} | {'Neural TM':^10} | {'Expected':^10} | {'Correct?':^10}")
            print("-" * 50)
            
            correct = 0
            for i, (s, pred, target) in enumerate(zip(used_test_strings, predictions, targets)):
                is_correct = (pred.item() > 0.5) == (target.item() > 0.5)
                correct += int(is_correct)
                print(f"{s:^10} | {pred.item():.4f}:^10 | {target.item():.0f}:^10 | {'✓' if is_correct else '✗':^10}")
            
            accuracy = correct / len(predictions)
            print("-" * 50)
            print(f"Overall accuracy: {accuracy:.2%}")
        
        # Interactive testing loop
        print("\n" + "-"*50)
        print("Interactive testing. Enter binary strings to test (empty line to quit).")
        print("The model will check if the string is a palindrome.")
        print("-"*50)
        
        while True:
            user_input = input("\nEnter a binary string (0s and 1s only): ")
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
                
            # Test with classical Turing Machine
            accepted, final_tape, steps = tm.process(user_input)
            print(f"\nClassical TM result: {'ACCEPT' if accepted else 'REJECT'} (in {steps} steps)")
            print(f"Final tape: {''.join(final_tape)}")
            
            # Test with Neural TM
            # One-hot encode the input
            input_tensor = torch.zeros(1, max(len(user_input), max_len), len(alphabet))
            for i, char in enumerate(user_input):
                input_tensor[0, i, char_to_idx.get(char, char_to_idx['_'])] = 1.0
                
            with torch.no_grad():
                output, _ = neural_tm(input_tensor)
                pred = (output > 0.5).float().item()
                prob = output.item()
                
            print(f"Neural TM result: {'ACCEPT' if prob > 0.5 else 'REJECT'} (confidence: {prob:.4f})")
            
            # Check if the string is actually a palindrome
            is_palindrome = user_input == user_input[::-1]
            print(f"Is actually a palindrome? {'Yes' if is_palindrome else 'No'}")
            print(f"A palindrome reads the same forwards and backwards (e.g., '101', '11', '000')")
                
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_turing_machine()
