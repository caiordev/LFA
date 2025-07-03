import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiniteAutomaton:
    """
    Implementation of a deterministic finite automaton (DFA).
    """
    def __init__(self, states, alphabet, transitions, initial_state, final_states):
        """
        Initialize a finite automaton.
        
        Args:
            states (list): List of states
            alphabet (list): List of symbols in the alphabet
            transitions (dict): Dictionary mapping (state, symbol) to next state
            initial_state: The initial state
            final_states (list): List of final/accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.final_states = final_states
        
    def process(self, input_string):
        """
        Process an input string and determine if it's accepted.
        
        Args:
            input_string (str): Input string to process
            
        Returns:
            bool: True if the string is accepted, False otherwise
        """
        current_state = self.initial_state
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                return False
                
            if (current_state, symbol) not in self.transitions:
                return False
                
            current_state = self.transitions[(current_state, symbol)]
            
        return current_state in self.final_states
        
    def to_transition_matrix(self):
        """
        Convert the automaton to a transition matrix representation.
        
        Returns:
            tuple: (transition_matrix, state_mapping, symbol_mapping)
                - transition_matrix: numpy array of shape (num_states, num_symbols)
                - state_mapping: dict mapping state to index
                - symbol_mapping: dict mapping symbol to index
        """
        state_mapping = {state: i for i, state in enumerate(self.states)}
        symbol_mapping = {symbol: i for i, symbol in enumerate(self.alphabet)}
        
        num_states = len(self.states)
        num_symbols = len(self.alphabet)
        
        # Initialize transition matrix with -1 (no transition)
        transition_matrix = np.full((num_states, num_symbols), -1, dtype=np.int32)
        
        # Fill in transitions
        for (state, symbol), next_state in self.transitions.items():
            state_idx = state_mapping[state]
            symbol_idx = symbol_mapping[symbol]
            next_state_idx = state_mapping[next_state]
            
            transition_matrix[state_idx, symbol_idx] = next_state_idx
            
        return transition_matrix, state_mapping, symbol_mapping
        
    def to_one_hot_encoding(self):
        """
        Convert the automaton to one-hot encoded tensors for neural network training.
        
        Returns:
            tuple: (transitions_tensor, initial_state_tensor, final_states_tensor)
                - transitions_tensor: tensor of shape (num_states, num_symbols, num_states)
                - initial_state_tensor: tensor of shape (num_states)
                - final_states_tensor: tensor of shape (num_states)
        """
        num_states = len(self.states)
        num_symbols = len(self.alphabet)
        
        state_mapping = {state: i for i, state in enumerate(self.states)}
        symbol_mapping = {symbol: i for i, symbol in enumerate(self.alphabet)}
        
        # Initialize tensors
        transitions_tensor = torch.zeros(num_states, num_symbols, num_states)
        initial_state_tensor = torch.zeros(num_states)
        final_states_tensor = torch.zeros(num_states)
        
        # Fill transitions tensor
        for (state, symbol), next_state in self.transitions.items():
            state_idx = state_mapping[state]
            symbol_idx = symbol_mapping[symbol]
            next_state_idx = state_mapping[next_state]
            
            transitions_tensor[state_idx, symbol_idx, next_state_idx] = 1.0
            
        # Set initial state
        initial_state_tensor[state_mapping[self.initial_state]] = 1.0
        
        # Set final states
        for state in self.final_states:
            final_states_tensor[state_mapping[state]] = 1.0
            
        return transitions_tensor, initial_state_tensor, final_states_tensor


class NeuralFiniteAutomaton(nn.Module):
    """
    Neural network implementation of a finite automaton.
    """
    def __init__(self, num_states, num_symbols):
        """
        Initialize a neural finite automaton.
        
        Args:
            num_states (int): Number of states
            num_symbols (int): Number of symbols in the alphabet
        """
        super(NeuralFiniteAutomaton, self).__init__()
        
        self.num_states = num_states
        self.num_symbols = num_symbols
        
        # Transition matrix: (num_states, num_symbols, num_states)
        self.transitions = nn.Parameter(torch.randn(num_states, num_symbols, num_states))
        
        # Initial state distribution: (num_states)
        self.initial_state = nn.Parameter(torch.randn(num_states))
        
        # Final state indicators: (num_states)
        self.final_states = nn.Parameter(torch.randn(num_states))
        
    def forward(self, input_sequence):
        """
        Process an input sequence.
        
        Args:
            input_sequence (torch.Tensor): One-hot encoded input sequence of shape (batch_size, seq_len, num_symbols)
            
        Returns:
            torch.Tensor: Acceptance probability for each sequence in the batch
        """
        batch_size, seq_len, _ = input_sequence.size()
        
        # Initialize state distribution with initial state
        state_dist = F.softmax(self.initial_state, dim=0).expand(batch_size, self.num_states)
        
        # Process each symbol in the sequence
        for t in range(seq_len):
            # Get symbol distribution at time t
            symbol_dist = input_sequence[:, t, :]  # (batch_size, num_symbols)
            
            # Compute next state distribution
            # state_dist: (batch_size, num_states)
            # symbol_dist: (batch_size, num_symbols)
            # transitions: (num_states, num_symbols, num_states)
            
            # Expand dimensions for broadcasting
            state_dist_expanded = state_dist.unsqueeze(2).unsqueeze(3)  # (batch_size, num_states, 1, 1)
            symbol_dist_expanded = symbol_dist.unsqueeze(1).unsqueeze(3)  # (batch_size, 1, num_symbols, 1)
            
            # Compute transition probabilities
            transition_probs = torch.softmax(self.transitions, dim=2)  # (num_states, num_symbols, num_states)
            transition_probs = transition_probs.unsqueeze(0)  # (1, num_states, num_symbols, num_states)
            
            # Combine state and symbol distributions with transitions
            next_state_dist = (state_dist_expanded * symbol_dist_expanded * transition_probs).sum(dim=(1, 2))
            
            # Update state distribution
            state_dist = next_state_dist
            
        # Compute acceptance probability using final state indicators
        final_state_probs = torch.sigmoid(self.final_states)
        acceptance_prob = (state_dist * final_state_probs.unsqueeze(0)).sum(dim=1)
        
        return acceptance_prob
        
    def from_automaton(self, automaton):
        """
        Initialize the neural automaton from a traditional automaton.
        
        Args:
            automaton (FiniteAutomaton): Traditional automaton
        """
        transitions_tensor, initial_state_tensor, final_states_tensor = automaton.to_one_hot_encoding()
        
        # Convert one-hot tensors to logits
        self.transitions.data = torch.log(transitions_tensor + 1e-10)
        self.initial_state.data = torch.log(initial_state_tensor + 1e-10)
        self.final_states.data = torch.log(final_states_tensor + 1e-10)
        
    def to_automaton(self, states, alphabet, threshold=0.5):
        """
        Convert the neural automaton to a traditional automaton.
        
        Args:
            states (list): List of state names
            alphabet (list): List of symbols in the alphabet
            threshold (float): Probability threshold for transitions and final states
            
        Returns:
            FiniteAutomaton: Traditional automaton
        """
        if len(states) != self.num_states or len(alphabet) != self.num_symbols:
            raise ValueError("Number of states or symbols doesn't match")
            
        # Get transition probabilities
        transition_probs = torch.softmax(self.transitions, dim=2).detach().numpy()
        
        # Get initial state distribution
        initial_state_probs = F.softmax(self.initial_state, dim=0).detach().numpy()
        
        # Get final state probabilities
        final_state_probs = torch.sigmoid(self.final_states).detach().numpy()
        
        # Determine initial state (highest probability)
        initial_state = states[np.argmax(initial_state_probs)]
        
        # Determine final states
        final_states = [states[i] for i in range(self.num_states) if final_state_probs[i] > threshold]
        
        # Determine transitions
        transitions = {}
        for i in range(self.num_states):
            for j in range(self.num_symbols):
                next_state_idx = np.argmax(transition_probs[i, j])
                if transition_probs[i, j, next_state_idx] > threshold:
                    transitions[(states[i], alphabet[j])] = states[next_state_idx]
                    
        return FiniteAutomaton(states, alphabet, transitions, initial_state, final_states)
