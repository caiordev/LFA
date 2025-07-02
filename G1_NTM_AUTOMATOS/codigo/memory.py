import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NTMMemory(nn.Module):
    """
    Neural Turing Machine Memory module.
    Implements a differentiable memory bank that can be read from and written to.
    """
    def __init__(self, N, M):
        """
        Initialize the NTM Memory module.
        
        Args:
            N (int): Number of memory locations (rows)
            M (int): Size of each memory location (columns)
        """
        super(NTMMemory, self).__init__()
        
        self.N = N  # Number of memory locations
        self.M = M  # Size of each memory location
        
        # Initialize memory with zeros
        self.register_buffer('memory', torch.zeros(N, M))
        
    def reset(self, batch_size=1):
        """
        Reset the memory to zeros.
        
        Args:
            batch_size (int): Batch size for parallel processing
        """
        self.memory = torch.zeros(batch_size, self.N, self.M, device=self.memory.device)
        
    def size(self):
        """
        Return the size of the memory.
        
        Returns:
            tuple: (N, M) - number of locations and size of each location
        """
        return self.N, self.M
        
    def read(self, weights):
        """
        Read from memory using attention weights.
        
        Args:
            weights (torch.Tensor): Attention weights of shape (batch_size, N)
            
        Returns:
            torch.Tensor: Read vectors of shape (batch_size, M)
        """
        # weights: (batch_size, N)
        # memory: (batch_size, N, M)
        # return: (batch_size, M)
        return torch.matmul(weights.unsqueeze(1), self.memory).squeeze(1)
        
    def write(self, weights, erase_vector, add_vector):
        """
        Write to memory using attention weights, erase and add vectors.
        
        Args:
            weights (torch.Tensor): Attention weights of shape (batch_size, N)
            erase_vector (torch.Tensor): Erase vector of shape (batch_size, M)
            add_vector (torch.Tensor): Add vector of shape (batch_size, M)
        """
        # weights: (batch_size, N)
        # erase_vector: (batch_size, M)
        # add_vector: (batch_size, M)
        
        # Expand dimensions for broadcasting
        weights = weights.unsqueeze(2)  # (batch_size, N, 1)
        erase_vector = erase_vector.unsqueeze(1)  # (batch_size, 1, M)
        add_vector = add_vector.unsqueeze(1)  # (batch_size, 1, M)
        
        # Erase operation
        erase = torch.matmul(weights, erase_vector)  # (batch_size, N, M)
        self.memory = self.memory * (1 - erase)
        
        # Add operation
        add = torch.matmul(weights, add_vector)  # (batch_size, N, M)
        self.memory = self.memory + add
        
    def address(self, key, beta, gate, shift, gamma, prev_weights):
        """
        Addressing mechanism for the memory.
        
        Args:
            key (torch.Tensor): Key vector of shape (batch_size, M)
            beta (torch.Tensor): Key strength scalar of shape (batch_size, 1)
            gate (torch.Tensor): Interpolation gate of shape (batch_size, 1)
            shift (torch.Tensor): Shift weighting of shape (batch_size, shift_range)
            gamma (torch.Tensor): Sharpening scalar of shape (batch_size, 1)
            prev_weights (torch.Tensor): Previous weights of shape (batch_size, N)
            
        Returns:
            torch.Tensor: New weights of shape (batch_size, N)
        """
        # Content addressing
        key = key.unsqueeze(1)  # (batch_size, 1, M)
        
        # Calculate cosine similarity
        memory_norm = F.normalize(self.memory, p=2, dim=2)
        key_norm = F.normalize(key, p=2, dim=2)
        similarity = torch.matmul(memory_norm, key_norm.transpose(1, 2)).squeeze(2)  # (batch_size, N)
        
        # Apply key strength (focus)
        content_weights = F.softmax(beta * similarity, dim=1)
        
        # Interpolation between previous and content weights
        interpolated_weights = gate * content_weights + (1 - gate) * prev_weights
        
        # Convolutional shift
        shift_weights = self._shift_operation(interpolated_weights, shift)
        
        # Sharpening
        weights = shift_weights ** gamma
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)
        
        return weights
        
    def _shift_operation(self, weights, shift):
        """
        Circular convolution operation for shifting attention.
        
        Args:
            weights (torch.Tensor): Weights of shape (batch_size, N)
            shift (torch.Tensor): Shift weighting of shape (batch_size, shift_range)
            
        Returns:
            torch.Tensor: Shifted weights of shape (batch_size, N)
        """
        batch_size = weights.size(0)
        N = self.N
        shift_range = shift.size(1)
        
        # Create a list of shifted weights
        shift_list = []
        for i in range(shift_range):
            shift_amount = i - shift_range // 2
            shift_list.append(torch.roll(weights, shifts=shift_amount, dims=1))
        
        # Stack and weight by the shift values
        shifted_weights = torch.stack(shift_list, dim=2)  # (batch_size, N, shift_range)
        result = torch.matmul(shifted_weights, shift.unsqueeze(2)).squeeze(2)  # (batch_size, N)
        
        return result
