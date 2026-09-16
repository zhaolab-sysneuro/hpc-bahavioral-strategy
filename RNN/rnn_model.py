import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from datetime import datetime
import random
import os

def gen_abc(num_samples, trial_len, width, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)  
    else:
        rng = np.random
    cue = {
        'S': torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
        'A': torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32),
        'B': torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32),
        'C': torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32),
        'E': torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32),
    }

    sequence = []
    num_blocks = num_samples//trial_len + 1
    width = width

    for _ in range(num_blocks):
        for _ in range(width):
            sequence.append(cue['S'])
        str = ['A', 'B', 'C']
        rng.shuffle(str)  
        for cue_key in str:
            for _ in range(width):
                sequence.append(cue[cue_key])
        for _ in range(width):
            sequence.append(cue['E'])

    return torch.stack(sequence[:num_samples])


def gen_r(seq):
    reward = {
        'R': torch.tensor([1, 0, 0], dtype=torch.float32),
        'N': torch.tensor([0, 1, 0], dtype=torch.float32),
        'ITI': torch.tensor([0, 0, 1], dtype=torch.float32),
    }
    return torch.stack([reward[r] for r in seq])


def save_model(model):

    date = datetime.now().strftime("%Y-%m-%d")
    base_dir = 'Summaries'
    date_dir = os.path.join(base_dir, date)
    os.makedirs(date_dir, exist_ok=True)

    model_files = [f for f in os.listdir(date_dir) if f.startswith("run") and f.endswith(".pth")]
    run_numbers = []
    
    for file in model_files:
        try:
            run_number = int(file.split("run")[1].split(".")[0])
            run_numbers.append(run_number)
        except:
            pass
    
    next_run_number = max(run_numbers) + 1 if run_numbers else 0
    model_filename = f"run{next_run_number}.pth"
    model_path = os.path.join(date_dir, model_filename)
    torch.save(model.state_dict(), model_path)
    print('model saved: ',model_path)

   
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, seed = None, **kwargs):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.6)

    def recurrence(self, input, hidden):
        h_new = nn.functional.relu(self.input2h(input) + self.h2h(hidden))   
        return h_new

    def forward(self, input, hidden=None):        
        if input.dim() == 2:
                    input = input.unsqueeze(1)
                    unsqueeze_input = True
        else:
            unsqueeze_input = False
        
        batch_size = input.shape[1]
        device = input.device
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size).to(device)
        elif hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
            if hidden.size(0) != batch_size:
                hidden = hidden.expand(batch_size, -1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            hidden = self.dropout(hidden)
            output.append(hidden)

        output = torch.stack(output, dim=0)

        if unsqueeze_input:
            output = output.squeeze(1)
            hidden = hidden.squeeze(0)
        return output, hidden

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seed = None, **kwargs):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.rnn = RNN(input_size, hidden_size, seed = seed, **kwargs)
        self.fc1 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden, _ = self.rnn(x) 
        output1 = self.fc1(hidden)
        return output1, hidden

