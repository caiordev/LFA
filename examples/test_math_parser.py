import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from formal_languages.math_parser import MathParser, NeuralMathParser
from ntm.ntm import NeuralTuringMachine

def test_math_parser():
    """
    Test the mathematical expression parser.
    """
    print("Testing mathematical expression parser...")
    
    # Create a math parser
    parser = MathParser()
    
    # Test expressions
    test_expressions = [
        "2+3",
        "5-2",
        "3*4",
        "10/2",
        "(2+3)*4",
        "2+(3*4)",
        "10/(2+3)",
        "2.5+3.5",
        "((2+3)*(4-1))/3"
    ]
    
    for expr in test_expressions:
        valid = parser.parse(expr)
        if valid:
            try:
                result = parser.evaluate(expr)
                print(f"Expression: {expr}, Result: {result}")
            except Exception as e:
                print(f"Expression: {expr}, Error: {e}")
        else:
            print(f"Expression: {expr}, Invalid syntax")
    
    print("\nTraining neural math parser on examples...")
    
    # ---------------- Generate synthetic training set (+ and - only) ----------------
    import random, operator, math
    ops = {'+':operator.add, '-':operator.sub}
    def random_expr(depth=0):
        if depth>2 or random.random()<0.4:
            return str(random.randint(1,9))
        left = random_expr(depth+1)
        right = random_expr(depth+1)
        op = random.choice(list(ops.keys()))
        if random.random()<0.5:
            return f"({left}{op}{right})"
        else:
            return f"{left}{op}{right}"
    def safe_eval(expr):
        try:
            return eval(expr)
        except ZeroDivisionError:
            return None
    examples=[]
    while len(examples)<5000:
        e=random_expr()
        val=safe_eval(e)
        if val is None or math.isinf(val) or math.isnan(val):
            continue
        examples.append((e,float(val)))
    # keep few predefined for later testing
    predefined=[("2+3",5.0),("5-2",3.0)]
    
    # Build alphabet from generated set (digits + plus/minus)
    alphabet=set('0123456789+-()')
    alphabet=sorted(alphabet)
    char_to_idx={c:i for i,c in enumerate(alphabet)}
    
    # Create parameters
    input_size = len(alphabet)
    controller_size = 64
    memory_size = 32
    memory_locations = 64
    hidden_size = 32
    output_size = 1
    
    # Create a simplified version for testing
    class NTMMathParser(nn.Module):
        def __init__(self, input_size, controller_size=64, memory_size=32, memory_locations=64, scale=1.0):
            super(NTMMathParser, self).__init__()
            self.scale=scale
            self.input_size=input_size
            self.ntm=NeuralTuringMachine(
                input_size=input_size,
                output_size=1,
                controller_size=controller_size,
                memory_size=memory_size,
                memory_locations=memory_locations,
            )
            
        def forward(self, x_onehot):
            # x_onehot: (seq_len, batch, input_size)
            seq_len,batch,_=x_onehot.shape
            outputs=self.ntm.forward_sequence(x_onehot)
            # take last output per sequence (last timestep)
            return outputs[-1] * self.scale
            
        @staticmethod
        def _partial_results(expr: str):
            """Return list of running totals for simple + - left-to-right."""
            total=0
            current=""
            op="+"
            partial=[]
            for ch in expr:
                if ch.isdigit():
                    current+=ch
                else: # operator
                    if current:
                        num=int(current)
                        total = total + num if op=='+' else total - num
                        current=""
                    op=ch
                # record after processing char
                partial.append(total)
            if current:
                num=int(current)
                total = total + num if op=='+' else total - num
            partial[-1]=total  # ensure last element final
            return partial

        def train_on_examples(self, examples, optimizer, num_epochs=50, batch_size=32):
            # Extract expressions and results
            expressions = [ex[0] for ex in examples]
            results = [ex[1] for ex in examples]
            
            # Create alphabet
            alphabet = set()
            for expr in expressions:
                alphabet.update(expr)
            alphabet = sorted(alphabet)
            char_to_idx = {char: i for i, char in enumerate(alphabet)}
            
                        # Precompute one-hot sequences and target sequences (partial sums)
            onehots=[]
            target_seqs=[]
            max_len=0
            for expr,val in examples:
                seq_len=len(expr)
                max_len=max(max_len,seq_len)
                oh=torch.zeros(seq_len,1,self.input_size)
                for t,ch in enumerate(expr):
                    oh[t,0,char_to_idx[ch]]=1.0
                onehots.append(oh)
                partial=self._partial_results(expr)
                target_seqs.append(torch.tensor(partial,dtype=torch.float32).unsqueeze(1))
            # pad to max_len
            for i in range(len(onehots)):
                pad_len=max_len-onehots[i].shape[0]
                if pad_len>0:
                    pad_x=torch.zeros(pad_len,1,self.input_size)
                    pad_val=target_seqs[i][-1].item() if isinstance(target_seqs[i][-1], torch.Tensor) else target_seqs[i][-1]
                    pad_y=torch.full((pad_len,1), pad_val)
                    onehots[i]=torch.cat([onehots[i],pad_x],dim=0)
                    target_seqs[i]=torch.cat([target_seqs[i],pad_y],dim=0)
            target_seqs=[t/self.scale for t in target_seqs]
                    
            # Normalize targets
            targets = torch.tensor(results, dtype=torch.float32).unsqueeze(1) / self.scale
            
            losses=[]
            criterion=nn.MSELoss()
            for epoch in range(num_epochs):
                perm=torch.randperm(len(examples))
                for i in range(0,len(examples),batch_size):
                    batch_idx=perm[i:i+batch_size]
                    optimizer.zero_grad()
                    outputs_batch=[]
                    target_batch=[]
                    for idx in batch_idx:
                        out_seq=self.ntm.forward_sequence(onehots[idx].to(next(self.parameters()).device))
                        outputs_batch.append(out_seq.squeeze(1).squeeze(-1))
                        target_batch.append(target_seqs[idx].to(out_seq.device).squeeze(1))
                    outputs_batch=torch.stack(outputs_batch)  # (batch,max_len)
                    target_batch=torch.stack(target_batch)
                    loss=criterion(outputs_batch,target_batch)
                    loss.backward()
                    optimizer.step()
                losses.append(loss.item())
                if (epoch+1)%5==0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
            return losses
            
        def evaluate_expression(self, expression, char_to_idx):
            seq_len=len(expression)
            x=torch.zeros(seq_len,1,self.input_size)
            for t,ch in enumerate(expression):
                x[t,0,char_to_idx[ch]]=1.0
            with torch.no_grad():
                out=self(x)
            return out.item()
    
    # Create Neural Math Parser using NTM
    neural_parser = NTMMathParser(input_size)
    
    # Create optimizer
    optimizer = optim.Adam(neural_parser.parameters(), lr=0.01)
    
    # Train the parser
    try:
        losses = neural_parser.train_on_examples(examples, optimizer, num_epochs=100)
        print(f"Final loss: {losses[-1]:.6f}")
        
        # Test the neural parser
        print("\nTesting neural math parser...")
        for expr, expected in examples[:5]:  # Test on first 5 examples
            result = neural_parser.evaluate_expression(expr, char_to_idx)
            print(f"Expression: {expr}, Neural Result: {result:.2f}, Expected: {expected}")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    test_math_parser()
