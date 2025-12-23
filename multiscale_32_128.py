import os
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure



# Set the seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)  # For Python random number generation
    np.random.seed(seed)  # For NumPy random number generation
    torch.manual_seed(seed)  # For PyTorch CPU
    torch.cuda.manual_seed(seed)  # For PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # For all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disabling this ensures deterministic results
    print(f"Seed set to {seed} for reproducibility")

# Call this function to set the seed before your training
set_seed(42)


base_dir = os.getcwd()  # current directory
save_dir = os.path.join(base_dir, 'Code')


def normalize_data(tensor):
    """Normalize tensor to the range [0, 1]."""
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor
    

def load_and_concatenate_data(main_dir, range_g, range_z, filename_template):
    """Load and concatenate data from multiple files."""
    concatenated_data = []

    for z in range_z:
        for g in range_g:
            # Construct the file name and path
            file_name = filename_template.format(g=g, z=z)
            file_path = os.path.join(main_dir, file_name)

            if os.path.exists(file_path):
                data = torch.load(file_path)
                print(f"Loaded data for g={g}, z={z}, shape: {data.shape}")

                # Normalize the data after loading
                data = normalize_data(data)

                concatenated_data.append(data)
            else:
                print(f"File not found: {file_path}")

    if concatenated_data:
        result = torch.cat(concatenated_data, dim=0)
        print(f"Concatenated data shape: {result.shape}")
    else:
        print("No data loaded for concatenation.")
        result = None

    return result

main_dir_input = os.path.join(base_dir, 'Data', 'Grid_32', 'Saturation')
main_dir_target = os.path.join(base_dir, 'Data', 'Grid_128', 'Saturation')



# Define ranges
range_g = [1006,1007,1008,1009]  # Extend this range if needed 1008,1009
range_z = [5]
filename_template_input = 'plume_1year_g{g}_z{z}.pt'
filename_template_target = 'plume_1year_g{g}_z{z}.pt'

# Load input and target data
concatenated_input = load_and_concatenate_data(main_dir_input, range_g, range_z, filename_template_input)
concatenated_target = load_and_concatenate_data(main_dir_target, range_g, range_z, filename_template_target)



print(f"Reordered input shape: {concatenated_input.shape}")
print(f"Reordered target shape: {concatenated_target.shape}")



def load_and_concatenate_permdata(main_dir_perm, range_g, range_z, epsilon=1e-8):
    # Initialize list to store loaded data tensors for concatenation
    concatenated_data = []

    # Loop over z values
    for z in range_z:
        # Loop over g values
        for g in range_g:
            # Construct the file name
            perm_file = f'g{g}_z{z}_all_perm_together.pt'
            # Construct the full path
            perm_path = os.path.join(main_dir_perm, perm_file)

            # Load the data if the file exists
            if os.path.exists(perm_path):
                data = torch.load(perm_path)
                print(f"Loaded data for g={g}, z={z}, shape: {data.shape}")

                # Apply log10 transformation with epsilon to avoid log10(0)
                data = torch.log10(data + epsilon)

                # Normalize the data to [0, 1]
                min_val = data.min()
                max_val = data.max()
                data = (data - min_val) / (max_val - min_val + epsilon)

                # Append the transformed data to the list
                concatenated_data.append(data)
            else:
                print(f"File not found: {perm_path}")

    # Check if any data was loaded
    if concatenated_data:
        # Concatenate the loaded tensors along the first dimension
        result = torch.cat(concatenated_data, dim=0)
        print(f"Concatenated data shape: {result.shape}")
    else:
        print("No data loaded for concatenation.")
        result = None

    return result

# Define directories
main_dir_perm = os.path.join(base_dir, 'Data', 'Grid_128', 'Permeability')


# Call the function to load and concatenate data
concatenated_perm = load_and_concatenate_permdata(main_dir_perm, range_g, range_z)


# Assuming concatenated_input and concatenated_target each have a size of [1000, ...]
train_size =800  # First 900 for training
val_size = 100
test_size = 100   # Last 100 for testing


#split validation
train_input_x = concatenated_input[:train_size]  # First 900 samples
val_input_x= concatenated_input[train_size:train_size+val_size]
test_input_x = concatenated_input[train_size+val_size:]   # Last 100 samples

# Split input
train_input_y = concatenated_perm[:train_size]  # First 900 samples
val_input_y= concatenated_perm[train_size:train_size+val_size]
test_input_y = concatenated_perm[train_size+val_size:]   # Last 100 samples

# Split target
train_target = concatenated_target[:train_size]  # First 900 samples
val_target= concatenated_target[train_size:train_size+val_size]
test_target = concatenated_target[train_size+val_size:]   # Last 100 samples



class FSRCNN3D(nn.Module):
    def __init__(self, d, p, m, input_channels=1, upscaling_factor_xy=2, upscaling_factor_z=2):
        
        super(FSRCNN3D, self).__init__()
        
       
        self.feature_extraction = nn.Conv3d(input_channels, d, kernel_size=5, padding='same')
        self.prelu1 = nn.PReLU()
        
       
        self.shrinking = nn.Conv3d(d, p, kernel_size=1, padding='same')
        self.prelu2 = nn.PReLU()
        
      
        self.mapping_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(p, p, kernel_size=3, padding=1),
                nn.PReLU()
            ) for _ in range(m)]
        )
        
       
        self.expanding = nn.Conv3d(p, d, kernel_size=1, padding=0)
        self.prelu3 = nn.PReLU()
        

        self.deconv = nn.ConvTranspose3d(d, 1, kernel_size=(2,8,8), 
                                         stride=(upscaling_factor_z, upscaling_factor_xy, upscaling_factor_xy), 
                                         padding=(2, 2, 2))

        self.nonmap1 = nn.Conv3d(2, d, kernel_size=(3, 3, 3), padding='same')
        self.nonmap2 = nn.Conv3d(d, p, kernel_size=(3, 3, 3), padding='same')
        self.nonmap3 = nn.Conv3d( p, 1, kernel_size=1, padding=0)
        

        
    def forward(self, x, y):

        x = self.prelu1(self.feature_extraction(x))
        
        x = self.prelu2(self.shrinking(x))
     
        x = self.mapping_layers(x)
    
        x = self.prelu3(self.expanding(x))
      
        x = self.prelu3(self.deconv(x))
    
        x = torch.cat((x, y), dim=1)

        x = self.prelu2(self.nonmap1(x))
      
        x = self.prelu2(self.nonmap2(x))
       
        x = self.nonmap3(x)
      
        return torch.sigmoid(x)  



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FSRCNN3D(d=32, m=4,p=16, input_channels=1, upscaling_factor_xy=4, upscaling_factor_z=4)
model.to(device)  

criterion = nn.L1Loss(reduction='sum')  # Use sum reduction
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Adjust learning rate every 10 epochs


def train_model(model, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for data_sat,data_perm, target in train_loader:
            data_sat,data_perm, target = data_sat.to(device),data_perm.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data_sat,data_perm)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            

        # Average training loss for this epoch
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for val_input_sat_batch,val_input_perm_batch, val_target_batch in val_loader:
                val_input_sat_batch,val_input_perm_batch, val_target_batch = val_input_sat_batch.to(device),val_input_perm_batch.to(device), val_target_batch.to(device)
                val_output = model( val_input_sat_batch,val_input_perm_batch,)
                val_loss = criterion(val_output, val_target_batch)
                total_val_loss += val_loss.item()
            
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step()  

    
    # Plot and save the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Epoch vs Loss')
    y_ticks = [10e-1, 10e0, 10e1, 10e2,10e3,10e4] 
    plt.yticks(y_ticks, labels=[str(ytick) for ytick in y_ticks])
    x_ticks = [50,100,200,300,400,500]  
    plt.xticks(x_ticks, labels=[str(xtick) for xtick in x_ticks])
    plt.legend()
    plt.grid()
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    print("Loss curves saved as epoch_vs_loss.png")

    return train_losses, val_losses



# Train the model
train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=500)
csv_file_path = os.path.join(save_dir, "loss_vs_epoch.csv")

def save_loss_to_csv(train_losses, val_losses, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])  # Header
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch, train_loss, val_loss])
    
    print(f"Loss data saved to: {csv_file_path}")

save_loss_to_csv(train_losses, val_losses, csv_file_path)

model_path = os.path.join(save_dir, "multiscale_32_128.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

