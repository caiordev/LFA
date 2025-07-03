import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from automata.finite_automaton import FiniteAutomaton

class RegexParser:
    """
    Parser for regular expressions.
    Converts regular expressions to finite automata.
    """
    def __init__(self):
        """
        Initialize the regex parser.
        """
        self.operators = {'|', '.', '*', '+', '?', '(', ')'}
        
    def parse(self, regex):
        """
        Parse a regular expression and convert it to a finite automaton.
        
        Args:
            regex (str): Regular expression string
            
        Returns:
            FiniteAutomaton: Finite automaton that recognizes the language defined by the regex
        """
        # Preprocess the regex to add explicit concatenation operators
        regex = self._add_concatenation(regex)
        
        # Convert to postfix notation
        postfix = self._to_postfix(regex)
        
        # Build NFA from postfix notation
        return self._build_automaton(postfix)
        
    def _add_concatenation(self, regex):
        """
        Add explicit concatenation operators to the regex.
        
        Args:
            regex (str): Regular expression string
            
        Returns:
            str: Regular expression with explicit concatenation operators
        """
        result = []
        for i in range(len(regex) - 1):
            result.append(regex[i])
            if (regex[i] not in '(|') and (regex[i+1] not in ')|*+?'):
                result.append('.')
        result.append(regex[-1])
        return ''.join(result)
        
    def _to_postfix(self, regex):
        """
        Convert infix regex to postfix notation using the Shunting Yard algorithm.
        
        Args:
            regex (str): Regular expression string in infix notation
            
        Returns:
            str: Regular expression in postfix notation
        """
        precedence = {'|': 1, '.': 2, '*': 3, '+': 3, '?': 3}
        stack = []
        output = []
        
        for char in regex:
            if char not in self.operators:
                output.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack and stack[-1] == '(':
                    stack.pop()  # Discard the '('
            else:
                while (stack and stack[-1] != '(' and 
                       stack[-1] in precedence and 
                       precedence.get(stack[-1], 0) >= precedence.get(char, 0)):
                    output.append(stack.pop())
                stack.append(char)
                
        while stack:
            output.append(stack.pop())
            
        return ''.join(output)
        
    def _build_automaton(self, postfix):
        """
        Build a finite automaton from a postfix regular expression.
        
        Args:
            postfix (str): Regular expression in postfix notation
            
        Returns:
            FiniteAutomaton: Finite automaton that recognizes the language defined by the regex
        """
        stack = []
        state_counter = 0
        
        for char in postfix:
            if char not in self.operators:
                # Create a simple NFA for a single character
                start_state = f'q{state_counter}'
                state_counter += 1
                accept_state = f'q{state_counter}'
                state_counter += 1
                
                states = {start_state, accept_state}
                alphabet = {char}
                transitions = {(start_state, char): accept_state}
                initial_state = start_state
                final_states = {accept_state}
                
                nfa = FiniteAutomaton(states, alphabet, transitions, initial_state, final_states)
                stack.append(nfa)
                
            elif char == '|':
                # Union operation
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                
                start_state = f'q{state_counter}'
                state_counter += 1
                accept_state = f'q{state_counter}'
                state_counter += 1
                
                states = {start_state, accept_state}.union(nfa1.states).union(nfa2.states)
                alphabet = nfa1.alphabet.union(nfa2.alphabet)
                
                transitions = {}
                transitions.update(nfa1.transitions)
                transitions.update(nfa2.transitions)
                transitions[(start_state, '')] = nfa1.initial_state
                transitions[(start_state, '')] = nfa2.initial_state
                
                for final_state in nfa1.final_states:
                    transitions[(final_state, '')] = accept_state
                    
                for final_state in nfa2.final_states:
                    transitions[(final_state, '')] = accept_state
                    
                nfa = FiniteAutomaton(states, alphabet, transitions, start_state, {accept_state})
                stack.append(nfa)
                
            elif char == '.':
                # Concatenation operation
                nfa2 = stack.pop()
                nfa1 = stack.pop()
                
                states = nfa1.states.union(nfa2.states)
                alphabet = nfa1.alphabet.union(nfa2.alphabet)
                
                transitions = {}
                transitions.update(nfa1.transitions)
                transitions.update(nfa2.transitions)
                
                for final_state in nfa1.final_states:
                    transitions[(final_state, '')] = nfa2.initial_state
                    
                nfa = FiniteAutomaton(states, alphabet, transitions, nfa1.initial_state, nfa2.final_states)
                stack.append(nfa)
                
            elif char == '*':
                # Kleene star operation
                nfa1 = stack.pop()
                
                start_state = f'q{state_counter}'
                state_counter += 1
                accept_state = f'q{state_counter}'
                state_counter += 1
                
                states = {start_state, accept_state}.union(nfa1.states)
                alphabet = nfa1.alphabet
                
                transitions = {}
                transitions.update(nfa1.transitions)
                transitions[(start_state, '')] = nfa1.initial_state
                transitions[(start_state, '')] = accept_state
                
                for final_state in nfa1.final_states:
                    transitions[(final_state, '')] = nfa1.initial_state
                    transitions[(final_state, '')] = accept_state
                    
                nfa = FiniteAutomaton(states, alphabet, transitions, start_state, {accept_state})
                stack.append(nfa)
                
            elif char == '+':
                # One or more operation
                nfa1 = stack.pop()
                
                start_state = f'q{state_counter}'
                state_counter += 1
                accept_state = f'q{state_counter}'
                state_counter += 1
                
                states = {start_state, accept_state}.union(nfa1.states)
                alphabet = nfa1.alphabet
                
                transitions = {}
                transitions.update(nfa1.transitions)
                transitions[(start_state, '')] = nfa1.initial_state
                
                for final_state in nfa1.final_states:
                    transitions[(final_state, '')] = nfa1.initial_state
                    transitions[(final_state, '')] = accept_state
                    
                nfa = FiniteAutomaton(states, alphabet, transitions, start_state, {accept_state})
                stack.append(nfa)
                
            elif char == '?':
                # Zero or one operation
                nfa1 = stack.pop()
                
                start_state = f'q{state_counter}'
                state_counter += 1
                accept_state = f'q{state_counter}'
                state_counter += 1
                
                states = {start_state, accept_state}.union(nfa1.states)
                alphabet = nfa1.alphabet
                
                transitions = {}
                transitions.update(nfa1.transitions)
                transitions[(start_state, '')] = nfa1.initial_state
                transitions[(start_state, '')] = accept_state
                
                for final_state in nfa1.final_states:
                    transitions[(final_state, '')] = accept_state
                    
                nfa = FiniteAutomaton(states, alphabet, transitions, start_state, {accept_state})
                stack.append(nfa)
                
        return stack.pop()


class NeuralRegexParser(nn.Module):
    """
    Neural network implementation of a regex parser.
    Uses a Neural Turing Machine to learn regex patterns.
    """
    def __init__(self, ntm, input_size, output_size):
        """
        Initialize the neural regex parser.
        
        Args:
            ntm: Neural Turing Machine instance
            input_size (int): Size of the input alphabet
            output_size (int): Size of the output (typically 1 for binary classification)
        """
        super(NeuralRegexParser, self).__init__()
        
        self.ntm = ntm
        self.input_size = input_size
        self.output_size = output_size
        
        # Input embedding - deve ser do tamanho do alfabeto
        self.input_embedding = nn.Embedding(input_size, input_size)
        
        # Output layer
        self.output_layer = nn.Linear(ntm.output_size, output_size)
        
    def forward(self, input_sequence):
        """
        Process an input sequence.
        
        Args:
            input_sequence (torch.Tensor): One-hot encoded input sequence of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = input_sequence.size()
        if seq_len == 0:
            # Retorna zero para sequências vazias
            return torch.zeros(batch_size, self.output_size, device=input_sequence.device)
        
        # Reset NTM
        self.ntm.reset(batch_size)
        
        # Process each symbol in the sequence
        outputs = []
        for t in range(seq_len):
            # Get input embedding
            input_emb = torch.matmul(input_sequence[:, t], self.input_embedding.weight)
            
            # Process through NTM
            ntm_output = self.ntm(input_emb)
            
            # Generate output
            output = self.output_layer(ntm_output)
            outputs.append(output)
            
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        # Return final output
        return outputs[:, -1]
    
    def train_on_examples(self, positive_examples, negative_examples, optimizer, num_epochs=100):
        """
        Train the neural regex parser on positive and negative examples.
        
        Args:
            positive_examples (list): List of strings that match the regex
            negative_examples (list): List of strings that don't match the regex
            optimizer: PyTorch optimizer
            num_epochs (int): Number of training epochs
            
        Returns:
            list: Training losses
        """
        # Convert examples to tensors
        all_examples = positive_examples + negative_examples
        all_labels = [1] * len(positive_examples) + [0] * len(negative_examples)
        
        # Create alphabet
        alphabet = set()
        for example in all_examples:
            alphabet.update(example)
        alphabet = sorted(alphabet)
        char_to_idx = {char: i for i, char in enumerate(alphabet)}
        
        # Convert examples to one-hot encoding
        max_len = max(len(example) for example in all_examples)
        inputs = torch.zeros(len(all_examples), max_len, len(alphabet))
        for i, example in enumerate(all_examples):
            for j, char in enumerate(example):
                inputs[i, j, char_to_idx[char]] = 1.0
                
        labels = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self(inputs)
            
            # Compute loss
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            losses.append(loss.item())
            
        return losses
