import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TuringMachine:
    """
    Implementation of a deterministic Turing machine.
    """
    def __init__(self, states, tape_alphabet, transitions, initial_state, blank_symbol, final_states):
        """
        Initialize a Turing machine.
        
        Args:
            states (list): List of states
            tape_alphabet (list): List of tape symbols
            transitions (dict): Dictionary mapping (state, symbol) to (next_state, write_symbol, move_direction)
                               where move_direction is 'L', 'R', or 'N' (left, right, or no move)
            initial_state: The initial state
            blank_symbol: The blank symbol used for empty tape cells
            final_states (list): List of final/accepting states
        """
        self.states = states
        self.tape_alphabet = tape_alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.blank_symbol = blank_symbol
        self.final_states = final_states
        
    def process(self, input_string, max_steps=10000):
        """
        Process an input string and determine if it's accepted.
        
        Args:
            input_string (str): Input string to process
            max_steps (int): Maximum number of steps to run before halting
            
        Returns:
            tuple: (accepted, tape, steps)
                - accepted: True if the string is accepted, False otherwise
                - tape: Final tape contents
                - steps: Number of steps executed
        """
        # Initialize tape with input string and blank symbols on both ends
        tape = list(input_string)
        
        # Initialize head position to the start of the tape
        head_pos = 0
        
        # Initialize current state
        current_state = self.initial_state
        
        # Run the machine
        steps = 0
        while current_state not in self.final_states and steps < max_steps:
            # Get current symbol
            if 0 <= head_pos < len(tape):
                current_symbol = tape[head_pos]
            else:
                # Extend tape if needed
                if head_pos < 0:
                    tape = [self.blank_symbol] * abs(head_pos) + tape
                    head_pos = 0
                else:  # head_pos >= len(tape)
                    tape = tape + [self.blank_symbol] * (head_pos - len(tape) + 1)
                current_symbol = self.blank_symbol
                
            # Check if there's a transition for the current state and symbol
            if (current_state, current_symbol) not in self.transitions:
                return False, tape, steps
                
            # Get next state, write symbol, and move direction
            next_state, write_symbol, move_direction = self.transitions[(current_state, current_symbol)]
            
            # Write symbol to tape
            tape[head_pos] = write_symbol
            
            # Move head
            if move_direction == 'L':
                head_pos -= 1
            elif move_direction == 'R':
                head_pos += 1
                
            # Update state
            current_state = next_state
            
            steps += 1
            
        # Check if we reached a final state
        accepted = current_state in self.final_states
        
        return accepted, tape, steps


class NeuralTuringMachineSimulator(nn.Module):
    """
    Neural network implementation of a Turing machine using a Neural Turing Machine.
    """
    def __init__(self, ntm, num_states, tape_alphabet_size):
        """
        Initialize a neural Turing machine simulator.
        
        Args:
            ntm: Neural Turing Machine instance
            num_states (int): Number of states
            tape_alphabet_size (int): Size of the tape alphabet
        """
        super(NeuralTuringMachineSimulator, self).__init__()
        
        self.ntm = ntm
        self.num_states = num_states
        self.tape_alphabet_size = tape_alphabet_size
        
        # State embedding
        self.state_embedding = nn.Embedding(num_states, ntm.controller_size // 2)
        
        # Tape symbol embedding
        self.symbol_embedding = nn.Embedding(tape_alphabet_size, ntm.controller_size // 2)
        
        # Output layers
        self.state_predictor = nn.Linear(ntm.output_size, num_states)
        self.write_symbol_predictor = nn.Linear(ntm.output_size, tape_alphabet_size)
        self.move_predictor = nn.Linear(ntm.output_size, 3)  # 3 directions: left, right, no move
        
        # Initial state
        self.initial_state = nn.Parameter(torch.zeros(num_states))
        
        # Final state indicators
        self.final_states = nn.Parameter(torch.zeros(num_states))
        
    def forward(self, input_sequence, max_steps=100):
        """
        Process an input sequence.
        
        Args:
            input_sequence (torch.Tensor): One-hot encoded input sequence of shape (batch_size, seq_len, tape_alphabet_size)
            max_steps (int): Maximum number of steps to run the Turing machine
            
        Returns:
            tuple: (acceptance_prob, final_tape)
                - acceptance_prob: Acceptance probability for each sequence in the batch
                - final_tape: Final tape contents for each sequence
        """
        batch_size, seq_len, _ = input_sequence.size()
        
        # Initialize state distribution with initial state
        state_dist = F.softmax(self.initial_state, dim=0).expand(batch_size, self.num_states)
        
        # Initialize tape with input sequence
        # We'll use the NTM's memory as the tape
        self.ntm.reset(batch_size)
        
        # Write input sequence to memory (sem inplace)
        memory = self.ntm.memory.memory.clone()
        for t in range(seq_len):
            # Get symbol embedding
            symbol_emb = torch.matmul(input_sequence[:, t], self.symbol_embedding.weight)
            
            # Write to memory at position t
            # This is a simplified approach; in a full implementation, we would use the NTM's write mechanism
            memory[:, t, :self.symbol_embedding.weight.size(1)] = symbol_emb
        self.ntm.memory.memory = memory
            
        # Initialize head position
        head_pos = torch.zeros(batch_size, dtype=torch.long)
        
        # Run the Turing machine for max_steps
        for step in range(max_steps):
            # Read current symbol from tape (memory)
            current_symbol = self.ntm.memory.read(F.one_hot(head_pos, self.ntm.memory_locations).float())
            
            # Get current state embedding
            state_emb = torch.matmul(state_dist, self.state_embedding.weight)
            
            # Combine state and symbol embeddings
            combined_input = torch.cat([state_emb, current_symbol[:, :self.symbol_embedding.weight.size(1)]], dim=1)
            
            # Process through NTM controller
            ntm_output = self.ntm(combined_input)
            
            # Predict next state
            next_state_logits = self.state_predictor(ntm_output)
            next_state_dist = F.softmax(next_state_logits, dim=1)
            # Predict write symbol
            write_symbol_logits = self.write_symbol_predictor(ntm_output)
            write_symbol_dist = F.softmax(write_symbol_logits, dim=1)
            write_symbol_emb = torch.matmul(write_symbol_dist, self.symbol_embedding.weight)
            # Predict move direction
            move_logits = self.move_predictor(ntm_output)
            move_dist = F.softmax(move_logits, dim=1)
            # Write symbol to tape (sem inplace)
            memory = self.ntm.memory.memory.clone()
            for b in range(batch_size):
                memory[b, head_pos[b], :self.symbol_embedding.weight.size(1)] = write_symbol_emb[b]
            self.ntm.memory.memory = memory
            # Move head
            move_direction = torch.argmax(move_dist, dim=1)  # 0: left, 1: right, 2: no move
            head_pos = head_pos + (move_direction == 1).long() - (move_direction == 0).long()
            # Ensure head position is within bounds
            head_pos = torch.clamp(head_pos, 0, self.ntm.memory_locations - 1)
            # Update state distribution
            state_dist = next_state_dist
            
        # Compute acceptance probability using final state indicators
        final_state_probs = torch.sigmoid(self.final_states)
        acceptance_prob = (state_dist * final_state_probs.unsqueeze(0)).sum(dim=1)
        
        # Read final tape contents
        final_tape = self.ntm.memory.memory.clone()
        
        return acceptance_prob, final_tape
