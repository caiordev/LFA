import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PushdownAutomaton:
    """
    Implementation of a pushdown automaton (PDA).
    """
    def __init__(self, states, input_alphabet, stack_alphabet, transitions, initial_state, initial_stack, final_states):
        """
        Initialize a pushdown automaton.
        
        Args:
            states (list): List of states
            input_alphabet (list): List of input symbols
            stack_alphabet (list): List of stack symbols
            transitions (dict): Dictionary mapping (state, input_symbol, stack_symbol) to list of (next_state, stack_push_symbols)
            initial_state: The initial state
            initial_stack: The initial stack (list of symbols)
            final_states (list): List of final/accepting states
        """
        self.states = states
        self.input_alphabet = input_alphabet
        self.stack_alphabet = stack_alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.initial_stack = initial_stack.copy() if isinstance(initial_stack, list) else [initial_stack]
        self.final_states = final_states
        
    def process(self, input_string):
        """
        Process an input string and determine if it's accepted.
        Uses a non-deterministic approach with backtracking.
        
        Args:
            input_string (str): Input string to process
            
        Returns:
            bool: True if the string is accepted, False otherwise
        """
        # Initial configuration
        configurations = [(self.initial_state, 0, self.initial_stack.copy())]
        
        while configurations:
            current_state, position, stack = configurations.pop()
            
            # Check if we've consumed all input and are in a final state
            if position == len(input_string) and current_state in self.final_states:
                return True
                
            # Check if we've consumed all input and can make epsilon transitions
            if position == len(input_string):
                # Try epsilon transitions
                if stack and (current_state, '', stack[-1]) in self.transitions:
                    for next_state, push_symbols in self.transitions[(current_state, '', stack[-1])]:
                        # Create a new stack by removing the top element and adding the push symbols
                        new_stack = stack[:-1].copy()  # Remove top element
                        # Add push symbols in reverse order
                        for symbol in reversed(push_symbols):
                            new_stack.append(symbol)
                        configurations.append((next_state, position, new_stack))
                continue
                
            # Get current input symbol
            symbol = input_string[position]
            
            # Check if symbol is valid
            if symbol not in self.input_alphabet:
                continue
                
            # Try transitions with current input symbol
            if stack and (current_state, symbol, stack[-1]) in self.transitions:
                for next_state, push_symbols in self.transitions[(current_state, symbol, stack[-1])]:
                    # Create a new stack by removing the top element and adding the push symbols
                    new_stack = stack[:-1].copy()  # Remove top element
                    # Add push symbols in reverse order
                    for symbol in reversed(push_symbols):
                        new_stack.append(symbol)
                    configurations.append((next_state, position + 1, new_stack))
                    
            # Try epsilon transitions
            if stack and (current_state, '', stack[-1]) in self.transitions:
                for next_state, push_symbols in self.transitions[(current_state, '', stack[-1])]:
                    # Create a new stack by removing the top element and adding the push symbols
                    new_stack = stack[:-1].copy()  # Remove top element
                    # Add push symbols in reverse order
                    for symbol in reversed(push_symbols):
                        new_stack.append(symbol)
                    configurations.append((next_state, position, new_stack))
                    
        return False


class NeuralPushdownAutomaton(nn.Module):
    """
    Neural network implementation of a pushdown automaton using a Neural Turing Machine.
    """
    def __init__(self, ntm, num_states, input_alphabet_size, stack_alphabet_size):
        """
        Initialize a neural pushdown automaton.
        
        Args:
            ntm: Neural Turing Machine instance
            num_states (int): Number of states
            input_alphabet_size (int): Size of the input alphabet
            stack_alphabet_size (int): Size of the stack alphabet
        """
        super(NeuralPushdownAutomaton, self).__init__()
        
        self.ntm = ntm
        self.num_states = num_states
        self.input_alphabet_size = input_alphabet_size
        self.stack_alphabet_size = stack_alphabet_size
        
        # State embedding
        self.state_embedding = nn.Embedding(num_states, ntm.controller_size // 2)
        
        # Input symbol embedding
        self.input_embedding = nn.Embedding(input_alphabet_size + 1, ntm.controller_size // 2)  # +1 for epsilon
        
        # Stack symbol embedding
        self.stack_embedding = nn.Embedding(stack_alphabet_size, ntm.memory_size)
        
        # Output layers
        self.state_predictor = nn.Linear(ntm.output_size, num_states)
        self.stack_action_predictor = nn.Linear(ntm.output_size, stack_alphabet_size + 1)  # +1 for pop action
        
        # Initial state and stack symbol
        self.initial_state = nn.Parameter(torch.zeros(num_states))
        self.initial_stack_symbol = nn.Parameter(torch.zeros(stack_alphabet_size))
        
        # Final state indicators
        self.final_states = nn.Parameter(torch.zeros(num_states))
        
    def forward(self, input_sequence, max_steps=100):
        """
        Process an input sequence.
        
        Args:
            input_sequence (torch.Tensor): One-hot encoded input sequence of shape (batch_size, seq_len, input_alphabet_size)
            max_steps (int): Maximum number of steps to run the automaton
            
        Returns:
            torch.Tensor: Acceptance probability for each sequence in the batch
        """
        batch_size, seq_len, _ = input_sequence.size()
        
        # Initialize state distribution with initial state
        state_dist = F.softmax(self.initial_state, dim=0).expand(batch_size, self.num_states)
        
        # Initialize stack with initial symbol
        stack_symbol_dist = F.softmax(self.initial_stack_symbol, dim=0)
        
        # Reset NTM
        self.ntm.reset(batch_size)
        
        # Write initial stack symbol to memory
        initial_stack_embedding = torch.matmul(stack_symbol_dist, self.stack_embedding.weight)
        
        # Process each input symbol
        position = torch.zeros(batch_size, dtype=torch.long)
        
        for step in range(max_steps):
            # Check if all sequences have been fully processed
            if (position >= seq_len).all():
                break
                
            # Get current input symbol for each sequence
            current_symbols = torch.zeros(batch_size, self.input_alphabet_size + 1)
            for b in range(batch_size):
                if position[b] < seq_len:
                    current_symbols[b] = F.pad(input_sequence[b, position[b]], (0, 1))
                else:
                    # Use epsilon for completed sequences
                    current_symbols[b, -1] = 1.0
                    
            # Get current state embedding
            state_emb = torch.matmul(state_dist, self.state_embedding.weight)
            
            # Get current input embedding
            input_emb = torch.matmul(current_symbols, self.input_embedding.weight)
            
            # Combine state and input embeddings
            combined_input = torch.cat([state_emb, input_emb], dim=1)
            
            # Process through NTM
            ntm_output = self.ntm(combined_input)
            
            # Predict next state
            next_state_logits = self.state_predictor(ntm_output)
            next_state_dist = F.softmax(next_state_logits, dim=1)
            
            # Predict stack action
            stack_action_logits = self.stack_action_predictor(ntm_output)
            stack_action_dist = F.softmax(stack_action_logits, dim=1)
            
            # Update state distribution
            state_dist = next_state_dist
            
            # Update position for sequences that made progress
            position = position + 1
            
        # Compute acceptance probability using final state indicators
        final_state_probs = torch.sigmoid(self.final_states)
        acceptance_prob = (state_dist * final_state_probs.unsqueeze(0)).sum(dim=1)
        
        return acceptance_prob
