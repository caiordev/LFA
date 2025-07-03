import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .memory import NTMMemory
from .controller import NTMController

class NeuralTuringMachine(nn.Module):
    """
    Neural Turing Machine implementation.
    Combines controller and memory to create a differentiable Turing machine.
    """
    def __init__(self, input_size, output_size, controller_size, memory_size, memory_locations, num_heads=2):
        """
        Initialize the Neural Turing Machine.
        
        Args:
            input_size (int): Size of the input vector
            output_size (int): Size of the output vector
            controller_size (int): Size of the controller hidden layer
            memory_size (int): Size of each memory location (M)
            memory_locations (int): Number of memory locations (N)
            num_heads (int): Number of read/write heads
        """
        super(NeuralTuringMachine, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.memory_size = memory_size
        self.memory_locations = memory_locations
        self.num_heads = num_heads
        # Learnable initial read vector
        self.init_read_vec = nn.Parameter(torch.zeros(num_heads, memory_size))
        
        # Create memory
        self.memory = NTMMemory(memory_locations, memory_size)
        
        # Create controller
        self.controller = NTMController(
            input_size=input_size,
            output_size=output_size,
            controller_size=controller_size,
            memory_size=memory_size,
            memory_locations=memory_locations,
            num_heads=num_heads
        )
        
        # Initialize read and write weights
        self.reset()
        
    def reset(self, batch_size=1):
        """
        Reset the NTM state.
        
        Args:
            batch_size (int): Batch size for parallel processing
        """
        # Reset memory
        self.memory.reset(batch_size)
        
        # Reset controller
        self.controller.reset(batch_size)
        
        # Initialize read and write weights
        device = next(self.parameters()).device
        
        self.read_weights = [
            torch.zeros(batch_size, self.memory_locations, device=device)
            for _ in range(self.num_heads)
        ]
        for rw in self.read_weights:
            rw[:, 0] = 1  # Initialize to read from the first location
            
        self.write_weights = [
            torch.zeros(batch_size, self.memory_locations, device=device)
            for _ in range(self.num_heads)
        ]
        for ww in self.write_weights:
            ww[:, 0] = 1  # Initialize to write to the first location
            
        self.read_vectors = [
            self.init_read_vec[i].unsqueeze(0).expand(batch_size, -1).to(device)
            for i in range(self.num_heads)
        ]
        
    def forward(self, x):
        """
        Forward pass through the NTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Get controller outputs and memory parameters
        output, read_params, write_params = self.controller(
            x, self.read_vectors, self.read_weights, self.write_weights
        )
        
        # Write to memory
        for head_idx in range(self.num_heads):
            write_key = write_params['keys'][:, head_idx]
            write_beta = write_params['betas'][:, head_idx]
            write_gate = write_params['gates'][:, head_idx]
            write_shift = write_params['shifts'][:, head_idx]
            write_gamma = write_params['gammas'][:, head_idx]
            erase_vector = write_params['erase'][:, head_idx]
            add_vector = write_params['add'][:, head_idx]
            
            # Update write weights
            self.write_weights[head_idx] = self.memory.address(
                write_key, write_beta, write_gate, write_shift, write_gamma,
                self.write_weights[head_idx]
            )
            
            # Write to memory
            self.memory.write(self.write_weights[head_idx], erase_vector, add_vector)
        
        # Read from memory
        for head_idx in range(self.num_heads):
            read_key = read_params['keys'][:, head_idx]
            read_beta = read_params['betas'][:, head_idx]
            read_gate = read_params['gates'][:, head_idx]
            read_shift = read_params['shifts'][:, head_idx]
            read_gamma = read_params['gammas'][:, head_idx]
            
            # Update read weights
            self.read_weights[head_idx] = self.memory.address(
                read_key, read_beta, read_gate, read_shift, read_gamma,
                self.read_weights[head_idx]
            )
            
            # Read from memory
            self.read_vectors[head_idx] = self.memory.read(self.read_weights[head_idx])
        
        return output
    
    def forward_sequence(self, x_sequence):
        """
        Process a sequence of inputs.
        
        Args:
            x_sequence (torch.Tensor): Input sequence of shape (seq_len, batch_size, input_size)
            
        Returns:
            torch.Tensor: Output sequence of shape (seq_len, batch_size, output_size)
        """
        seq_len, batch_size, _ = x_sequence.size()
        
        # Reset NTM state
        self.reset(batch_size)
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            output = self(x_sequence[t])
            outputs.append(output)
            
        return torch.stack(outputs)
