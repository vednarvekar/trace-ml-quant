import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import MultiTimeframeCNN

# ------ LOAD DATA -------
print("Loading Master Data...")
X1 = torch.from_numpy(np.load("data/master_training/MASTER_X1.npy")).float().unsqueeze(1)
X5 = torch.from_numpy(np.load("data/master_training/MASTER_X5.npy")).float().unsqueeze(1)
XH = torch.from_numpy(np.load("data/master_training/MASTER_XH.npy")).float().unsqueeze(1)
Y  = torch.from_numpy(np.load("data/master_training/MASTER_y.npy")).long()

dataset = TensorDataset(X1, X5, XH, Y)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ------ SETUP MODEL ------
device = torch.device("cpu")
model = MultiTimeframeCNN().to(device)

# CRITICAL: WEIGHTS
# Give Neutral (0) a weight of 1.0, and Buy/Sell (1, 2) a weight of 10.0
# This stops the "Guessing Neutral" behavior.
weights = torch.tensor([1.0, 10.0, 10.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# --- TRAINING ---
print(f"Starting Training on {device}...")
model.train()

# --- IMPROVED SETUP ---
# Added Weight Decay (L2 Regularization) to force generalization
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

# Added Scheduler: Reduces LR by half if the loss doesn't improve for 2 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# --- IMPROVED TRAINING ---
for epoch in range(30):
    model.train() # Ensure dropout is active
    total_loss = 0
    
    for b_x1, b_x5, b_xh, b_y in train_loader:
        b_x1, b_x5, b_xh, b_y = b_x1.to(device), b_x5.to(device), b_xh.to(device), b_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(b_x1, b_x5, b_xh)
        
        # Loss uses the weights you defined to punish "Guessing Neutral"
        loss = criterion(outputs, b_y) 
        loss.backward()
        
        # Gradient Clipping: Prevents sudden "explosions" in data from breaking the model
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss/len(train_loader)
    scheduler.step(avg_loss) # Tell the scheduler the current loss
    
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")

# Save the Pattern Master
torch.save(model.state_dict(), "models/pattern_master_cnn.pth")
print("Model Saved Successfully.")