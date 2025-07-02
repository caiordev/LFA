import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NTMController(nn.Module):
    """
    Neural Turing Machine Controller module.
    Implements a neural network controller that interfaces with the memory.
    """
    def __init__(self, input_size, output_size, controller_size, memory_size, memory_locations, num_heads=1):
        """
        Initialize the NTM Controller.
        
        Args:
            input_size (int): Size of the input vector
            output_size (int): Size of the output vector
            controller_size (int): Size of the controller hidden layer
            memory_size (int): Size of each memory location (M)
            memory_locations (int): Number of memory locations (N)
            num_heads (int): Number of read/write heads
        """
        super(NTMController, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.controller_size = controller_size
        self.memory_size = memory_size
        self.memory_locations = memory_locations
        self.num_heads = num_heads
        
        # Feed-forward controller: single hidden layer
        self.fc = nn.Sequential(
            nn.Linear(input_size + num_heads * memory_size, controller_size),
            nn.ReLU()
        )
        
        # Output layer
        self.output = nn.Linear(controller_size + num_heads * memory_size, output_size)
        
        # Memory parameters for each head
        # Read heads
        self.read_keys = nn.Linear(controller_size, num_heads * memory_size)
        self.read_betas = nn.Linear(controller_size, num_heads)  # Key strength
        self.read_gates = nn.Linear(controller_size, num_heads)  # Interpolation gate
        self.read_shifts = nn.Linear(controller_size, num_heads * 3)  # Shift weighting (3 positions: -1, 0, 1)
        self.read_gammas = nn.Linear(controller_size, num_heads)  # Sharpening
        
        # Write heads
        self.write_keys = nn.Linear(controller_size, num_heads * memory_size)
        self.write_betas = nn.Linear(controller_size, num_heads)
        self.write_gates = nn.Linear(controller_size, num_heads)
        self.write_shifts = nn.Linear(controller_size, num_heads * 3)
        self.write_gammas = nn.Linear(controller_size, num_heads)
        
        self.erase_vectors = nn.Linear(controller_size, num_heads * memory_size)
        self.add_vectors = nn.Linear(controller_size, num_heads * memory_size)
        
        # No recurrent hidden state needed for feed-forward controller
        self.reset()
        
    def reset(self, batch_size=1):
        """
        Reset the controller state.
        
        Args:
            batch_size (int): Batch size for parallel processing
        """
        device = next(self.parameters()).device
        self.hidden = torch.zeros(batch_size, self.controller_size, device=device)
        
    def forward(self, x, prev_read_vectors, prev_read_weights, prev_write_weights):
        """
        Forward pass through the controller.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            prev_read_vectors (list): List of previous read vectors, each of shape (batch_size, memory_size)
            prev_read_weights (list): List of previous read weights, each of shape (batch_size, memory_locations)
            prev_write_weights (list): List of previous write weights, each of shape (batch_size, memory_locations)
            
        Returns:
            tuple: (output, read_params, write_params)
                - output: Output tensor of shape (batch_size, output_size)
                - read_params: Parameters for reading from memory
                - write_params: Parameters for writing to memory
        """
        batch_size = x.size(0)
        
        # Concatenate input with read vectors
        controller_input = torch.cat([x] + prev_read_vectors, dim=1)
        
        # Update controller state
        # Feed-forward: compute hidden directly from current input and previous read vectors
        self.hidden = self.fc(controller_input)
        
        # Generate parameters for memory interaction
        read_params = self._get_read_params(self.hidden)
        write_params = self._get_write_params(self.hidden)
        
        # Generate output
        controller_output = torch.cat([self.hidden] + prev_read_vectors, dim=1)
        output = self.output(controller_output)
        
        return output, read_params, write_params
    
    def _get_read_params(self, hidden):
        """
        Generate parameters for reading from memory.
        
        Args:
            hidden (torch.Tensor): Hidden state of shape (batch_size, controller_size)
            
        Returns:
            dict: Dictionary containing read parameters
        """
        batch_size = hidden.size(0)
        
        # Generate read parameters
        read_keys = self.read_keys(hidden).view(batch_size, self.num_heads, self.memory_size)
        read_betas = F.softplus(self.read_betas(hidden)).view(batch_size, self.num_heads, 1)
        read_gates = torch.sigmoid(self.read_gates(hidden)).view(batch_size, self.num_heads, 1)
        read_shifts = F.softmax(self.read_shifts(hidden).view(batch_size, self.num_heads, 3), dim=2)
        read_gammas = 1 + F.softplus(self.read_gammas(hidden)).view(batch_size, self.num_heads, 1)
        
        return {
            'keys': read_keys,
            'betas': read_betas,
            'gates': read_gates,
            'shifts': read_shifts,
            'gammas': read_gammas
        }
    
    def _get_write_params(self, hidden):
        """
        Generate parameters for writing to memory.
        
        Args:
            hidden (torch.Tensor): Hidden state of shape (batch_size, controller_size)
            
        Returns:
            dict: Dictionary containing write parameters
        """
        batch_size = hidden.size(0)
        
        # Generate write parameters
        write_keys = self.write_keys(hidden).view(batch_size, self.num_heads, self.memory_size)
        write_betas = F.softplus(self.write_betas(hidden)).view(batch_size, self.num_heads, 1)
        write_gates = torch.sigmoid(self.write_gates(hidden)).view(batch_size, self.num_heads, 1)
        write_shifts = F.softmax(self.write_shifts(hidden).view(batch_size, self.num_heads, 3), dim=2)
        write_gammas = 1 + F.softplus(self.write_gammas(hidden)).view(batch_size, self.num_heads, 1)
        
        # Clip erase (0.1–0.9) and scale add (|add|≤0.5) to stabilise training
        erase_vectors = torch.clamp(torch.sigmoid(self.erase_vectors(hidden)), 0.1, 0.9)
        erase_vectors = erase_vectors.view(batch_size, self.num_heads, self.memory_size)
        add_vectors = 0.5 * torch.tanh(self.add_vectors(hidden))
        add_vectors = add_vectors.view(batch_size, self.num_heads, self.memory_size)
        
        return {
            'keys': write_keys,
            'betas': write_betas,
            'gates': write_gates,
            'shifts': write_shifts,
            'gammas': write_gammas,
            'erase': erase_vectors,
            'add': add_vectors
        }
