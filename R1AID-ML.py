# R1AID-ML.py
# This script loads data, splits it, conducts 5-fold CV, and then tests on a hold-out set.
# It implements "SMoLK" by Sully Chen
#Import libraries
from tqdm import tqdm
import pickle
import numpy as np
import random
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import pandas as pd
from itertools import product
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.utils import resample
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from datetime import datetime

#Environment setup
device = "cuda"
USE_SMOTE = True

#Random setup
seed = 15
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Load data
with open(r"C:\Users\tjz5\Desktop\1-Projects\Blink-Protected\Code\DF-EPZSc.pkl", "rb") as f:
    dataDF = pickle.load(f)

#Drop any NaN
dataDF = dataDF.dropna(subset=['DataNorm', 'CaseID', 'SignalQuality']).reset_index(drop=True)

#Setup X and patientID to stratify
X = []
patientID = []
for i in range(len(dataDF)):
    d = dataDF['DataNorm'][i] 
    dvals = d
    X.append(dvals)
    patientID.append(dataDF['CaseID'][i])

#Convert to numpy arrays
X = np.array(X)
y = dataDF['SignalQuality'].astype(int)
singal_map = {1:0,2:1,3:0,4:0,5:0} #Map to binary classification (1 for R1, 0 for Non-R1)
y = y.map(singal_map)
y = np.array(y)

#Compute power spectra
PowerSpectra = []
for i in tqdm(range(0, len(X))):
    PowerSpectra.append(periodogram(X[i], fs=12800)[1])
PowerSpectra = np.float32(PowerSpectra)

#Convert patientID to numpy array and get unique patients
patientID = np.array(patientID)
unique_patients = np.unique(patientID)

#Convert to PyTorch tensors and move to device
X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
PowerSpectra = torch.tensor(PowerSpectra, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.long).to(device)

#Shuffle patients to ensure randomness
random.shuffle(unique_patients)

#Calculate samples per patient
samples_per_patient = []
for p in unique_patients:
    samples_per_patient.append(np.sum(patientID == p))
print(np.mean(samples_per_patient), np.std(samples_per_patient))
print(np.min(samples_per_patient), np.max(samples_per_patient))

#SMoLK Setup
LARGE_KERNEL_SIZE = 32
MEDIUM_KERNEL_SIZE = 16
SMALL_KERNEL_SIZE = 4
SPECTRA_SIZE = 225

#SMoLK Class
class LearnedFilters(nn.Module):
    def __init__(self, num_kernels=24, num_classes=4):
        super(LearnedFilters, self).__init__()
        self.conv1 = nn.Conv1d(1, num_kernels, LARGE_KERNEL_SIZE, stride=1, bias=True)
        self.conv2 = nn.Conv1d(1, num_kernels, MEDIUM_KERNEL_SIZE, stride=1, bias=True)
        self.conv3 = nn.Conv1d(1, num_kernels, SMALL_KERNEL_SIZE, stride=1, bias=True)
        
        self.linear = nn.Linear(num_kernels*3 + SPECTRA_SIZE, num_classes)
    
    def forward(self, x, powerspectrum):
        c1 = F.leaky_relu(self.conv1(x)).mean(dim=-1)
        c2 = F.leaky_relu(self.conv2(x)).mean(dim=-1)
        c3 = F.leaky_relu(self.conv3(x)).mean(dim=-1)
        
        aggregate = torch.cat([c1,c2,c3, powerspectrum], dim=1)
        aggregate = self.linear(aggregate)
        
        return aggregate

#Create patient mapping for train/test splits    
patient_mapping = {k: [] for k in unique_patients}

for i in range(0, len(patientID)):
    patient_mapping[patientID[i].item()].append(i)


def create_stratified_split(unique_patients, y_all, patientID_all, train_pct=0.75):
    """
    Create a stratified train/test split based on patient outcomes
    Args:
        unique_patients (np.ndarray): Array of unique patient IDs.
        y_all (torch.Tensor): Tensor of labels for all samples.
        patientID_all (np.ndarray): Array of patient IDs corresponding to each sample.
        train_pct (float): Proportion of data to use for training.
    Returns:
        tuple: Two lists containing patient IDs for training and testing sets.
    """
    # Calculate outcome ratio per patient
    patient_outcomes = {}
    for patient in unique_patients:
        patient_mask = patientID_all == patient
        patient_labels = y_all[patient_mask]
        patient_outcomes[patient] = np.mean(patient_labels.cpu().numpy())
    
    # Sort patients by outcome ratio for stratification
    sorted_patients = sorted(patient_outcomes.keys(), key=lambda x: patient_outcomes[x])
    
    # Create stratified split
    n_train_patients = int(len(unique_patients) * train_pct)
    
    # Take every nth patient to maintain stratification
    stride = len(unique_patients) / n_train_patients
    train_patient_indices = [int(i * stride) for i in range(n_train_patients)]
    train_patients = [sorted_patients[i] for i in train_patient_indices]
    test_patients = [p for p in unique_patients if p not in train_patients]
    
    return train_patients, test_patients

# Create the 75/25 split
train_patients, test_patients = create_stratified_split(unique_patients, y, patientID)

# Print train/test patient counts
print(f"Train patients: {len(train_patients)}")
print(f"Test patients: {len(test_patients)}")

# Get indices for train and test sets
train_indices_main = []
test_indices_main = []

for patient in train_patients:
    train_indices_main.extend(patient_mapping[patient])
    
for patient in test_patients:
    test_indices_main.extend(patient_mapping[patient])

#Print sample counts
print(f"Train samples: {len(train_indices_main)}")
print(f"Test samples: {len(test_indices_main)}")

# Create patient mapping for training patients only for CV
train_patient_mapping = {k: patient_mapping[k] for k in train_patients}

def generate_cv_split(fold, total_folds, train_patients_list):
    """
    Generate CV splits within the training patients
    Args:
        fold (int): Current fold index (0 to total_folds-1).
        total_folds (int): Total number of folds for CV.
        train_patients_list (list): List of training patient IDs.
    Returns:
        tuple: Two lists containing indices for training and validation sets.
    """
    cv_train_indices = []
    cv_val_indices = []
    
    for i, patient in enumerate(train_patients_list):
        if i % total_folds == fold:
            cv_val_indices.extend(train_patient_mapping[patient])
        else:
            cv_train_indices.extend(train_patient_mapping[patient])
    
    return cv_train_indices, cv_val_indices

#Define parameters
param_grid = {
    'num_kernels': [32],
    'lr': [1e-3],
    'weight_decay': [1e-5],
    'num_steps': [2048],
    'max_lr': [1e-2]
}

#5-Fold CV 
cv_folds = 5
best_params = None
best_cv_score = -np.inf
cv_results = []

print(f"Total parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")

# Generate all parameter combinations
param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())
fold_preds = []
fold_trues = []
for param_idx, param_values in enumerate(tqdm(param_combinations, desc="Parameter combinations")):
    # Create parameter dictionary
    params = dict(zip(param_names, param_values))
    
    fold_scores = []
    
    for fold in range(cv_folds):
        # Get CV train/val split
        cv_train_idx, cv_val_idx = generate_cv_split(fold, cv_folds, train_patients)
        
        # Prepare data
        X_cv_train = X[cv_train_idx]
        X_cv_val = X[cv_val_idx]
        PowerSpectra_cv_train = PowerSpectra[cv_train_idx]
        PowerSpectra_cv_val = PowerSpectra[cv_val_idx]
        y_cv_train = y[cv_train_idx]
        y_cv_val = y[cv_val_idx]
        
        # Apply SMOTE 
        if USE_SMOTE:
            sm = SMOTE(random_state=seed)
            X_cv_train_np, y_cv_train_np = sm.fit_resample(
                X_cv_train.cpu().numpy().reshape(-1, X_cv_train.shape[-1]), 
                y_cv_train.cpu().numpy()
            )
            X_cv_train = torch.tensor(X_cv_train_np, dtype=torch.float32).unsqueeze(1).to(device)
            y_cv_train = torch.tensor(y_cv_train_np, dtype=torch.long).to(device)
            
            # Recompute power spectra for SMOTE samples
            PowerSpectra_cv_train_list = []
            for i in range(X_cv_train.shape[0]):
                PowerSpectra_cv_train_list.append(periodogram(X_cv_train[i].cpu().numpy().squeeze(), fs=12800)[1])
            PowerSpectra_cv_train = torch.tensor(np.float32(PowerSpectra_cv_train_list), dtype=torch.float32).to(device)
        
        # Calculate class weights
        weights = np.bincount(y_cv_train.cpu().numpy())
        class_weights = torch.tensor(weights.sum() / (len(weights) * weights), dtype=torch.float32).to(device)
        
        # Initialize model with current parameters
        model = LearnedFilters(num_kernels=params['num_kernels'], num_classes=2).to(device)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params['max_lr'],
            total_steps=params['num_steps'],
            pct_start=0.05,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95
        )
        
        # Training loop
        model.train()
        for step in range(params['num_steps']):
            optimizer.zero_grad()
            out = model(X_cv_train, PowerSpectra_cv_train)
            loss = F.cross_entropy(out, y_cv_train, weight=class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            y_val_prob = model(X_cv_val, PowerSpectra_cv_val).softmax(dim=-1)
            fold_preds.append(y_val_prob.cpu().numpy())
            fold_trues.append(y_cv_val.cpu().numpy())
            val_auc = roc_auc_score(y_cv_val.cpu().numpy(), y_val_prob.cpu().numpy()[:, -1])
            fold_scores.append(val_auc)
    
    # Calculate mean CV score for this parameter combination
    mean_cv_score = np.mean(fold_scores)
    std_cv_score = np.std(fold_scores)
    
    # Store results
    cv_results.append({
        'params': params.copy(),
        'mean_cv_score': mean_cv_score,
        'std_cv_score': std_cv_score,
        'fold_scores': fold_scores.copy()
    })
    
    # Update best parameters
    if mean_cv_score > best_cv_score:
        best_cv_score = mean_cv_score
        best_params = params.copy()
    
    # Print progress for top combinations
    if param_idx < 10 or mean_cv_score > best_cv_score - 0.01:
        print(f"Params {param_idx+1}: {params}")
        print(f"CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

print(f"\nBest CV Score: {best_cv_score:.4f}")
print(f"Best Parameters: {best_params}")




# Retrain on full training set with best parameters
print("\nRetraining on full training set with best parameters...")

# Prepare full training data
X_train_full = X[train_indices_main]
PowerSpectra_train_full = PowerSpectra[train_indices_main]
y_train_full = y[train_indices_main]

# Apply SMOTE to full training set
if USE_SMOTE:
    sm = SMOTE(random_state=seed)
    X_train_full_np, y_train_full_np = sm.fit_resample(
        X_train_full.cpu().numpy().reshape(-1, X_train_full.shape[-1]), 
        y_train_full.cpu().numpy()
    )
    X_train_full = torch.tensor(X_train_full_np, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_full = torch.tensor(y_train_full_np, dtype=torch.long).to(device)
    
    # Recompute power spectra
    PowerSpectra_train_full_list = []
    for i in range(X_train_full.shape[0]):
        PowerSpectra_train_full_list.append(periodogram(X_train_full[i].cpu().numpy().squeeze(), fs=12800)[1])
    PowerSpectra_train_full = torch.tensor(np.float32(PowerSpectra_train_full_list), dtype=torch.float32).to(device)

# Calculate class weights
weights = np.bincount(y_train_full.cpu().numpy())
class_weights = torch.tensor(weights.sum() / (len(weights) * weights), dtype=torch.float32).to(device)

# Initialize final model with best parameters
final_model = LearnedFilters(num_kernels=best_params['num_kernels'], num_classes=2).to(device)

# Initialize optimizer and scheduler with best parameters
optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=best_params['max_lr'],
    total_steps=best_params['num_steps'],
    pct_start=0.05,
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95
)

# Prepare test data
X_test_final = X[test_indices_main]
PowerSpectra_test_final = PowerSpectra[test_indices_main]
y_test_final = y[test_indices_main]

# Training loop with monitoring
train_losses_final = []
train_aucs_final = []
train_f1s_final = []
train_accs_final = []

# Final training loop
pbar = tqdm(range(best_params['num_steps']), desc="Final training")
for step in pbar:
    # Training
    final_model.train()
    optimizer.zero_grad()
    out = final_model(X_train_full, PowerSpectra_train_full)
    loss = F.cross_entropy(out, y_train_full, weight=class_weights)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_losses_final.append(loss.item())
    
    # Update progress bar
    if step % 50 == 0 or step == best_params['num_steps'] - 1:
        final_model.eval()
        with torch.no_grad():
            y_train_prob = final_model(X_train_full, PowerSpectra_train_full).softmax(dim=-1)
            y_train_pred = torch.argmax(y_train_prob, dim=1)
            train_auc = roc_auc_score(y_train_full.cpu().numpy(), y_train_prob.cpu().numpy()[:, -1])
            train_f1 = f1_score(y_train_full.cpu().numpy(), y_train_pred.cpu().numpy())
            train_acc = accuracy_score(y_train_full.cpu().numpy(), y_train_pred.cpu().numpy())
            
            train_aucs_final.append(train_auc)
            train_f1s_final.append(train_f1)
            train_accs_final.append(train_acc)
            
            pbar.set_description(f"Training - Loss: {loss:.5f}, Train AUC: {train_auc:.4f}, F1: {train_f1:.4f}, Acc: {train_acc*100:.1f}%")
    else:
        # Just show loss for other steps
        pbar.set_description(f"Training - Loss: {loss:.5f}")

#Test set evaluation
print("\nEvaluating final model on held-out test set...")
final_model.eval()
with torch.no_grad():
    y_test_prob_final = final_model(X_test_final, PowerSpectra_test_final).softmax(dim=-1)
    y_test_pred_final = torch.argmax(y_test_prob_final, dim=1)
    final_test_auc = roc_auc_score(y_test_final.cpu().numpy(), y_test_prob_final.cpu().numpy()[:, -1])
    final_test_f1 = f1_score(y_test_final.cpu().numpy(), y_test_pred_final.cpu().numpy())
    final_test_acc = accuracy_score(y_test_final.cpu().numpy(), y_test_pred_final.cpu().numpy())

#Save model
torch.save(final_model, 'final_model.pth')
#save all needed variables
with open('cv_results.pkl', 'wb') as f:
    pickle.dump(cv_results, f)

with open('cv-ensemble_preds.pkl', 'wb') as f:
    pickle.dump({'fold_trues': fold_trues, 'fold_preds': fold_preds}, f)

with open('final_test_results.pkl', 'wb') as f:
    pickle.dump({
        'y_test_true': y_test_final.cpu().numpy(),
        'y_test_pred': y_test_pred_final.cpu().numpy(),
        'y_test_prob': y_test_prob_final.cpu().numpy(),
    }, f)
with open('train_indices_main.pkl', 'wb') as f:
    pickle.dump(train_indices_main, f)
with open('test_indices_main.pkl', 'wb') as f:
    pickle.dump(test_indices_main, f)
with open('X.pkl', 'wb') as f:
    pickle.dump(X.cpu().numpy(), f)
with open('y.pkl', 'wb') as f:
    pickle.dump(y.cpu().numpy(), f)
#These pickles can be used in the 'ModelAnalysis.py' script to generate performance statistics and plots
#---------------------------------------------------------------------------------
#Code for Figure 3b, generate and save random learned kenel weight plots
#Interpretability analysis
COI = 1 #Class of Interest, 1=R1

X = X.to(device)
y = y.to(device)
PowerSpectra = PowerSpectra.to(device)

indices = np.arange(len(X))[y.cpu()==COI]
idx = np.random.choice(indices)
print(idx)
print(y[idx].item())

num_kernels = 32
net = final_model.to(device)
sd = final_model.state_dict()

x = torch.tensor(X[idx], dtype=torch.float32, device=device).unsqueeze(0)
#time this next line
start = datetime.now()

pred = F.softmax(net(x, PowerSpectra[idx].unsqueeze(0)), dim=-1)
end = datetime.now()
time_taken = end - start
print('Time: ',time_taken) 
print(pred.detach().cpu().numpy())

c1 = F.leaky_relu(net.conv1(x)[0])
c2 = F.leaky_relu(net.conv2(x)[0])
c3 = F.leaky_relu(net.conv3(x)[0])

x = x[0, 0]
feature_contribution = (torch.cat([c1.mean(dim=-1),c2.mean(dim=-1),c3.mean(dim=-1)]) * sd['linear.weight'][COI, :num_kernels*3]).sum().item()
spectral_contribution = np.sum(PowerSpectra[idx].cpu().numpy() * sd['linear.weight'][COI, -SPECTRA_SIZE:].detach().cpu().numpy())

with torch.no_grad():
    accum = torch.zeros_like(x)
    for i in range(num_kernels):
        for j in range(x.shape[-1] - LARGE_KERNEL_SIZE + 1):
            accum[j:j+LARGE_KERNEL_SIZE] += c1[i, j] * sd['linear.weight'][COI, i] / LARGE_KERNEL_SIZE
        for j in range(x.shape[-1] - MEDIUM_KERNEL_SIZE + 1):
            accum[j:j+MEDIUM_KERNEL_SIZE] += c2[i, j] * sd['linear.weight'][COI, i+num_kernels] / MEDIUM_KERNEL_SIZE
        for j in range(x.shape[-1] - SMALL_KERNEL_SIZE + 1):
            accum[j:j+SMALL_KERNEL_SIZE] += c3[i, j] * sd['linear.weight'][COI, i+num_kernels*2] / SMALL_KERNEL_SIZE

accum = accum.cpu().numpy()[LARGE_KERNEL_SIZE//2:-LARGE_KERNEL_SIZE//2]
x = x.cpu().numpy()[LARGE_KERNEL_SIZE//2:-LARGE_KERNEL_SIZE//2]

plt.figure(figsize=(5, 1), dpi=300)
norm = plt.Normalize(accum.min(), accum.max())

num_interpolated_points = len(x) * 100  # 100 times the number of original points
linspace = np.linspace(0, len(x)-1, num_interpolated_points)
accum_interp = np.interp(linspace, np.arange(len(x)), accum)
point_sizes = np.interp(np.abs(accum_interp), (np.abs(accum_interp).min(), np.abs(accum_interp).max()), (10, 100))
x_interp = np.interp(linspace, np.arange(0, len(x)), x)

# Choose a suitable colormap (you can use any other colormap as needed)
colormap = plt.get_cmap('RdBu_r')

# Plot the curve
plt.plot(np.arange(0, len(x)), x, c='k', linewidth=0.5)  # Using black curve for better visualization

# Overlay with colored points
point_sizes = np.interp(np.abs(accum_interp), (np.abs(accum_interp).min(), np.abs(accum_interp).max()), (0.01, 0.5))
plt.scatter(linspace, x_interp, c=accum_interp, cmap=colormap, norm=norm, s=point_sizes)

# Show colorbar
cbar = plt.colorbar()

# Remove axis ticks
plt.xticks([])
plt.yticks([])
plt.xlim(0, len(x)-1)

plt.show()

#Visualize frequency spectrum 
plt.figure(figsize=(5, 2), dpi=150)
plt.title(f"Frequency Spectrum, Contribution: {spectral_contribution:.2f}")

weight = sd['linear.weight'][COI, -SPECTRA_SIZE:].cpu().numpy() * PowerSpectra[idx].cpu().numpy()

# Normalize weight for color mapping
norm = plt.Normalize(weight.min(), weight.max())
colormap = plt.get_cmap('RdBu_r')
colors = colormap(norm(weight))

plt.bar(np.linspace(0, LARGE_KERNEL_SIZE, len(PowerSpectra[idx])), PowerSpectra[idx].cpu(), color=colors, width=1/len(PowerSpectra[idx])*LARGE_KERNEL_SIZE*2)
plt.xlim(0, 5)
plt.xlabel("Frequency (Hz)")

#