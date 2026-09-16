import torch
import torch.nn as nn
from rnn_model import RNNNet, gen_abc, gen_r
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
device = torch.device('cuda')
print(f"Using device: {device}")

def gen_sequence(trials=10, trial_len=5, batch=32, width=1, delta_pred=1, seed=None, device='cpu'):

    num_trials = batch * trials
    total_len = num_trials * trial_len + delta_pred
    sequence = gen_abc(total_len, trial_len, width=width, seed=seed)
    cue_dict = {0: 'S', 1: 'A', 2: 'B', 3: 'C', 4: 'E'}
    reward_dict = {'S': 'N', 'A': 'R', 'B': 'N', 'C': 'R', 'E': 'N'}

    argmax_sequence = torch.argmax(sequence, dim=1).tolist()
    cue = [cue_dict[s] for s in argmax_sequence]
    reward = [reward_dict[c] for c in cue]
    sequence_reward = gen_r(reward)
    
    seq_len = total_len - delta_pred

    x = sequence[:seq_len]  
    x_next = sequence[delta_pred:delta_pred+seq_len] 
    
    r = sequence_reward[:seq_len]  
    r_next = sequence_reward[delta_pred:delta_pred+seq_len]  
    
    pos_list = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
    pos_num = torch.tensor(pos_list, device=device)
    
    pos_num_expanded = pos_num.repeat(num_trials)[:seq_len]
    pos_size = trial_len
    pos = torch.zeros(seq_len, pos_size, dtype=torch.float32, device=device)
    pos.scatter_(1, pos_num_expanded.unsqueeze(1), 1)  
    
    num_samples = trial_len * trials
    x = x.view(batch, num_samples, 5)
    x_next = x_next.view(batch, num_samples, 5)
    r = r.view(batch, num_samples, 3)
    r_next = r_next.view(batch, num_samples, 3)
    pos = pos.view(batch, num_samples, pos_size)
    
    return x, x_next, r, r_next, pos

def calculate_accuracy(pred, target):

    if target.dim() == 3: 
        target = torch.argmax(target, dim=-1)

    if pred.dim() == 3:
        pred_indices = torch.argmax(pred, dim=-1)  
    else:
        pred_indices = torch.argmax(pred, dim=1) 

    correct = (pred_indices == target).float().mean().item()
    return correct

def classify_trials(input_x, hidden_states, n_timesteps=5, cues = [1, 2, 3]):
    n_timesteps = n_timesteps
    num_trials = len(input_x)//n_timesteps
    input_states = input_x[:num_trials*n_timesteps].reshape(-1, n_timesteps)
    hidden_states = hidden_states[:num_trials*n_timesteps, :]
    n_sequences = input_states.shape[0]
    num_cells = hidden_states.shape[1]
    sequence_ids = np.repeat(range(n_sequences), n_timesteps)
    timestep_ids = np.tile(range(n_timesteps), n_sequences)
    
    df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'timestep': timestep_ids,
        **{f'hidden_{i}': hidden_states[:, i] for i in range(num_cells)}
    })
    
    type_mapping = {
        '123': 'ABC', '132': 'ACB', '213': 'BAC', 
        '231': 'BCA', '312': 'CAB', '321': 'CBA'
    }
    
    sequence_types = []
    for i in range(n_sequences):
        cues = cues
        pattern = ''.join(map(str, input_states[i, cues].astype(int)))
        seq_type = type_mapping.get(pattern, 'Unknown')
        sequence_types.append(seq_type)
    
    df['sequence_type'] = np.repeat(sequence_types, n_timesteps)
    
    grouped = df.groupby(['sequence_type', 'timestep']).mean()
    
    type_order = ['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA']
    data_type_list = []
    
    for type_name in type_order:
        for timestep in range(n_timesteps):
            if (type_name, timestep) in grouped.index:
                row_data = grouped.loc[(type_name, timestep)].values
                data_type_list.append(row_data)
            else:
                data_type_list.append(np.zeros(num_cells))
    
    data_type = np.array(data_type_list)
    
    return data_type[:, 1:], type_order

batch_size = 64
trials = 20
delta_pred = 1
width = 1
n_timesteps = 5
reward_zone = [[1], [2], [3]]
cues = [reward_zone[0][0], reward_zone[1][0], reward_zone[2][0]]

N = 100

data_mat_all = np.zeros((N, 6, n_timesteps, 100))
rsa_euclidean_all = np.zeros((N, 11, 11))
rsa_cosine_all = np.zeros((N, 11, 11))

states_euclidean_all = np.zeros((N, 9, 9))
states_cosine_all = np.zeros((N, 9, 9))

for seed in range(100):
    print(f'{seed}/100')
    np.random.seed(seed)

    losses = []

    class Dataset(TensorDataset):
        def __init__(self, x, x_next, r, r_next, pos):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(x_next, torch.Tensor):
                x_next = torch.tensor(x_next, dtype=torch.float32)
            if not isinstance(r, torch.Tensor):
                r = torch.tensor(r, dtype=torch.float32)
            if not isinstance(r_next, torch.Tensor):
                r_next = torch.tensor(r_next, dtype=torch.float32)
            if not isinstance(pos, torch.Tensor):
                pos = torch.tensor(pos, dtype=torch.float32)
            
            super().__init__(x, x_next, r, r_next, pos)
        
        def __getitem__(self, idx):
            return (self.tensors[0][idx],  # x
                    self.tensors[1][idx],  # x_next
                    self.tensors[2][idx],  # r
                    self.tensors[3][idx],  # r_next
                    self.tensors[4][idx])  # pos
        
    x, x_next, r, r_next, position = gen_sequence(trials=trials, trial_len=n_timesteps, width=width, batch=batch_size, delta_pred=delta_pred)
    dataset = Dataset(x, x_next, r, r_next, position)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

    x, x_next, r, r_next, position = gen_sequence(trials=trials, trial_len=n_timesteps, width=width, batch=batch_size, delta_pred=delta_pred)
    dataset = Dataset(x, x_next, r, r_next, position)
    val_loader = DataLoader(dataset, batch_size = batch_size, shuffle=False)

    load_model = False
    model = RNNNet(input_size=5, hidden_size=100, output_size=6, seed = seed) 
    if load_model:
        model.load_state_dict(torch.load("/Backup1/LYR/2024_2025/RNN/Summaries/2025-10-28/run10.pth", map_location = device))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 3000

    train_acc1_history = []
    train_acc2_history = []
    val_acc1_history = []
    val_acc2_history = []

    if not load_model:
        for epoch in range(epochs):
            model.train()
            epoch_acc1 = 0.0
            epoch_acc2 = 0.0
            batch_count = 0
            for batch in train_loader:
                input_x, _, target_curr, target_next, pos = batch
                input_x = input_x.permute(1, 0, 2).to(device)
                target_next = target_next.permute(1, 0, 2).to(device)
                target_curr = target_curr.permute(1, 0, 2).to(device)
                pred, hidden = model(input_x)
                loss1 = criterion(pred[..., :3].reshape(-1, 3), torch.argmax(target_next, dim=-1).reshape(-1))
                loss2 = criterion(pred[..., 3:].reshape(-1, 3), torch.argmax(target_curr, dim=-1).reshape(-1))
                loss = loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_acc1 = calculate_accuracy(pred[..., :3], target_next)
                epoch_acc1 += batch_acc1
                batch_acc2 = calculate_accuracy(pred[..., 3:], target_curr)
                epoch_acc2 += batch_acc2
                batch_count += 1

            epoch_acc1 /= batch_count
            train_acc1_history.append(epoch_acc1)
            epoch_acc2 /= batch_count
            train_acc2_history.append(epoch_acc2)
            model.eval()
            val_acc1 = 0.0
            val_acc2 = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_x, _, target_curr, target_next, pos = batch
                    input_x = input_x.permute(1, 0, 2).to(device)
                    target_next = target_next.permute(1, 0, 2).to(device)
                    target_curr = target_curr.permute(1, 0, 2).to(device)
                    pred, _ = model(input_x)
                    batch_acc1 = calculate_accuracy(pred[..., :3], target_next)
                    val_acc1 += batch_acc1
                    batch_acc2 = calculate_accuracy(pred[..., 3:], target_curr)
                    val_acc2 += batch_acc2
                    val_batch_count += 1
            
            val_acc1 /= val_batch_count
            val_acc1_history.append(val_acc1)
            val_acc2 /= val_batch_count
            val_acc2_history.append(val_acc2)


    x, x_next, r, r_next, pos = gen_sequence(trials=1000, trial_len=n_timesteps, width=width, batch=1, delta_pred=delta_pred, seed=seed)
    model.eval()
    with torch.no_grad():
        x = x.squeeze().to(device)
        _, hidden = model(x)

    hidden_states = hidden.cpu().numpy()
    input_x = np.argmax(x.cpu().numpy(), 1)

    data_type, type_order = classify_trials(input_x, hidden_states, n_timesteps = n_timesteps, cues = cues)

    data_mat = data_type.reshape(6, n_timesteps, -1)
    data_mat_all[seed] = data_mat

    num_cells = data_mat.shape[-1]

    # data_mat: ABC ACB BAC BCA CAB CBA
    data_h = np.zeros((11, num_cells))
    #R1A1 R1C1
    data_h[0, :] = np.mean(data_mat[[0, 1], :, :][:, reward_zone[0], :], axis=(0, 1)) # ABC ACB
    data_h[1, :] = np.mean(data_mat[[5, 4], :, :][:, reward_zone[0], :], axis=(0, 1)) # CBA CAB
    #R1A2 R1C2
    data_h[2, :] = np.mean(data_mat[[2], :, :][:, reward_zone[1], :], axis=(0, 1)) # BAC
    data_h[3, :] = np.mean(data_mat[[3], :, :][:, reward_zone[1], :], axis=(0, 1)) # BCA
    #R2A2 R2C2
    data_h[4, :] = np.mean(data_mat[[4], :, :][:, reward_zone[1], :], axis=(0, 1)) # CAB
    data_h[5, :] = np.mean(data_mat[[1], :, :][:, reward_zone[1], :], axis=(0, 1)) # ACB
    #R2A3 R2C3
    data_h[6, :] = np.mean(data_mat[[3, 5], :, :][:, reward_zone[2], :], axis=(0, 1)) # BCA CBA
    data_h[7, :] = np.mean(data_mat[[2, 0], :, :][:, reward_zone[2], :], axis=(0, 1)) # BAC ABC
    #B1
    data_h[8, :] = np.mean(data_mat[[1, 4], :, :][:, reward_zone[0], :], axis=(0, 1)) # BAC BCA
    #B2
    data_h[9, :] = np.mean(data_mat[[2, 3], :, :][:, reward_zone[1], :], axis=(0, 1)) # ABC CBA
    #B3
    data_h[10, :] = np.mean(data_mat[[0, 5], :, :][:, reward_zone[2], :], axis=(0, 1)) # ACB CAB

    # data_h = (data_h - np.mean(data_h))/np.std(data_h)
    dist_euclidean = cdist(data_h, data_h, metric='euclidean')
    # dist_cosine = cdist(data_h, data_h, metric='cosine')
    corr_coef = np.corrcoef(data_h)
    rsa_euclidean_all[seed] = dist_euclidean
    rsa_cosine_all[seed] = corr_coef

    num_cells = data_mat.shape[-1]

    # data_mat: ABC ACB BAC BCA CAB CBA
    data_h = np.zeros((9, num_cells))
    #RRB for 1st R
    data_h[0, :] = np.mean(data_mat[[1, 4], :, :][:, reward_zone[0], :], axis=(0, 1)) # ACB CAB
    #RBR for 1st R
    data_h[1, :] = np.mean(data_mat[[0, 5], :, :][:, reward_zone[0], :], axis=(0, 1)) # ABC CBA
    #BRR for 1st R
    data_h[2, :] = np.mean(data_mat[[2, 3], :, :][:, reward_zone[1], :], axis=(0, 1)) # BAC BCA
    #RRB for 2nd R
    data_h[3, :] = np.mean(data_mat[[1, 4], :, :][:, reward_zone[1], :], axis=(0, 1)) # ACB CAB
    #RBR for 2nd R
    data_h[4, :] = np.mean(data_mat[[0, 5], :, :][:, reward_zone[2], :], axis=(0, 1)) # ABC CBA
    #BRR for 2nd R
    data_h[5, :] = np.mean(data_mat[[2, 3], :, :][:, reward_zone[2], :], axis=(0, 1)) # BAC BCA
    # RRB for B
    data_h[6, :] = np.mean(data_mat[[1, 4], :, :][:, reward_zone[2], :], axis=(0, 1)) # ACB CAB
    # RBR for B
    data_h[7, :] = np.mean(data_mat[[0, 5], :, :][:, reward_zone[1], :], axis=(0, 1)) # ABC CBA
    # BRR for B
    data_h[8, :] = np.mean(data_mat[[2, 3], :, :][:, reward_zone[0], :], axis=(0, 1)) # BAC BCA

    data_h = (data_h - np.mean(data_h, 0))/np.std(data_h)
    dist_euclidean = cdist(data_h, data_h, metric='euclidean')
    dist_cosine = cdist(data_h, data_h, metric='correlation')
    corr_coef = np.corrcoef(data_h)
    states_euclidean_all[seed] = dist_euclidean
    states_cosine_all[seed] = corr_coef

np.save('../plot/data/rnn_rsa_euclidean_all.npy', rsa_euclidean_all)
np.save('../plot/data/rnn_rsa_cosine_all.npy', rsa_cosine_all)
np.save('../plot/data/rnn_states_euclidean_all.npy', states_euclidean_all)
np.save('../plot/data/rnn_states_cosine_all.npy', states_cosine_all)
np.save('../plot/data/rnn_data_mat_all.npy', data_mat_all)