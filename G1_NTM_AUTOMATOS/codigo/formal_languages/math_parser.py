import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from automata.pushdown_automaton import PushdownAutomaton

class MathParser:
    """
    Parser for mathematical expressions.
    Uses a pushdown automaton to parse and evaluate expressions.
    """
    def __init__(self):
        """
        Initialize the math parser.
        """
        # For simplicity, we'll use a direct approach for parsing
        # rather than a full PDA implementation
        self.operators = {'+', '-', '*', '/'}
        self.parentheses = {'(', ')'}
        self.digits = set('0123456789.')
        self.valid_chars = self.operators | self.parentheses | self.digits
        
    def parse(self, expression):
        """
        Parse and validate a mathematical expression.
        
        Args:
            expression (str): Mathematical expression string
            
        Returns:
            bool: True if the expression is valid, False otherwise
        """
        # Remove whitespace
        expression = expression.replace(' ', '')
        
        # Check if all characters are valid
        if not all(c in self.valid_chars for c in expression):
            return False
            
        # Check for balanced parentheses
        stack = []
        for c in expression:
            if c == '(':
                stack.append(c)
            elif c == ')':
                if not stack or stack[-1] != '(':
                    return False
                stack.pop()
        if stack:  # Unbalanced parentheses
            return False
            
        # Check for valid operator usage
        prev_char = None
        for i, c in enumerate(expression):
            # No operator at the beginning or end
            if c in self.operators and (i == 0 or i == len(expression) - 1):
                return False
            # No consecutive operators
            if c in self.operators and prev_char in self.operators:
                return False
            # No operator after opening parenthesis or before closing parenthesis
            if c in self.operators and prev_char == '(':
                return False
            if c == ')' and prev_char in self.operators:
                return False
            prev_char = c
            
        # Try to evaluate the expression (final validation)
        try:
            eval(expression)
            return True
        except:
            return False
        
    def evaluate(self, expression):
        """
        Evaluate a mathematical expression.
        
        Args:
            expression (str): Mathematical expression string
            
        Returns:
            float: Result of the evaluation
        """
        # Remove whitespace
        expression = expression.replace(' ', '')
        
        # Check if the expression is valid
        if not self.parse(expression):
            raise ValueError("Invalid expression")
            
        # Use Python's eval for simplicity
        # In a real implementation, you would implement a proper evaluator
        return eval(expression)


class NeuralMathParser(nn.Module):
    """
    Neural network implementation of a math expression parser.
    Uses a Neural Turing Machine to parse and evaluate expressions.
    """
    def __init__(self, ntm, input_size, hidden_size, output_size):
        """
        Initialize the neural math parser.
        
        Args:
            ntm: Neural Turing Machine instance
            input_size (int): Size of the input alphabet
            hidden_size (int): Size of the hidden layer
            output_size (int): Size of the output (typically 1 for numerical result)
        """
        super(NeuralMathParser, self).__init__()
        
        self.ntm = ntm
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Ensure NTM input size matches our input size
        assert ntm.input_size == input_size, f"NTM input size ({ntm.input_size}) must match parser input size ({input_size})"
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,  # Use input_size directly
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        """
        Process an input sequence.
        
        Args:
            input_sequence (torch.Tensor): One-hot encoded input sequence of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = input_sequence.size()
        
        # Reset NTM
        self.ntm.reset(batch_size)
        
        # Process through LSTM directly (no embedding needed since input is one-hot)
        lstm_out, _ = self.lstm(input_sequence)
        
        # Get the final LSTM output
        final_lstm_out = lstm_out[:, -1]
        
        # Process through NTM
        ntm_out = self.ntm(final_lstm_out)
        
        # Use the final output
        final_output = self.output_layer(ntm_out)
        
        return final_output
    
    def train_on_examples(self, examples, optimizer, num_epochs=100):
        """
        Train the neural math parser on examples.
        
        Args:
            examples (list): List of tuples (expression, result)
            optimizer: PyTorch optimizer
            num_epochs (int): Number of training epochs
            
        Returns:
            list: Training losses
        """
        # Extract expressions and results
        expressions = [ex[0] for ex in examples]
        results = [ex[1] for ex in examples]
        
        # Create alphabet
        alphabet = set()
        for expr in expressions:
            alphabet.update(expr)
        alphabet = sorted(alphabet)
        char_to_idx = {char: i for i, char in enumerate(alphabet)}
        
        # Convert expressions to one-hot encoding
        max_len = max(len(expr) for expr in expressions)
        inputs = torch.zeros(len(expressions), max_len, len(alphabet))
        for i, expr in enumerate(expressions):
            for j, char in enumerate(expr):
                inputs[i, j, char_to_idx[char]] = 1.0
                
        # Convert results to tensors
        targets = torch.tensor(results, dtype=torch.float32).unsqueeze(1)
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            
            # Compute loss
            loss = F.mse_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            losses.append(loss.item())
            
        return losses
    
    def evaluate_expression(self, expression, char_to_idx):
        """
        Evaluate a mathematical expression.
        
        Args:
            expression (str): Mathematical expression string
            char_to_idx (dict): Mapping from characters to indices
            
        Returns:
            float: Result of the evaluation
        """
        # Convert expression to one-hot encoding
        input_tensor = torch.zeros(1, len(expression), len(char_to_idx))
        for i, char in enumerate(expression):
            if char in char_to_idx:
                input_tensor[0, i, char_to_idx[char]] = 1.0
                
        # Forward pass
        with torch.no_grad():
            output = self(input_tensor)
            
        return output.item()
