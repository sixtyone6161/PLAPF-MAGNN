import os
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool
from torch.nn import Linear
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from model import PAIGN
from datetime import datetime


embedding_path = './save_tempt/ATOMTYPE/combined.json'
training_set_path = './save_tempt/gign_data/gign_training_set'
validation_set_path = './save_tempt/gign_data/gign_validation_set'


class ProteinLigandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, embedding_path):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.json')]

        with open(embedding_path, 'r') as f:
            self.embeddings = json.load(f)

        self.ligand_atom_type_map = self.embeddings['ligand']['atom_type']
        self.ligand_residue_map = self.embeddings['ligand']['residue']
        self.protein_atom_type_map = self.embeddings['protein']['atom_name']
        self.protein_residue_map = self.embeddings['protein']['residue_name']

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_names[idx])

        with open(file_path, 'r') as f:
            data = json.load(f)


        ligand_atoms = data['ligand']['atoms']
        ligand_coords = [[atom['x'], atom['y'], atom['z']] for atom in ligand_atoms]
        ligand_atom_types = [self.ligand_atom_type_map[atom['atom_type']] for atom in ligand_atoms]
        ligand_residues = [self.ligand_residue_map[atom['residue']] for atom in ligand_atoms]
        ligand_partial_charges = [atom['partial_charge'] for atom in ligand_atoms]
        ligand_is_polar = [atom['is_polar'] for atom in ligand_atoms]
        ligand_edge_index = data['ligand']['edge_index']

        protein_atoms = data['protein']['protein_atoms']
        protein_coords = [[atom['x'], atom['y'], atom['z']] for atom in protein_atoms]
        protein_atom_names = [self.protein_atom_type_map[atom['atom_name']] for atom in protein_atoms]
        protein_residue_names = [self.protein_residue_map[atom['residue_name']] for atom in protein_atoms]
        protein_is_polar = [atom['is_polar'] for atom in protein_atoms]
        protein_edge_index = data['protein']['edge_index']

        interaction_name = data['interaction_name']
        affinity = data['affinity']

        ligand_coords = torch.tensor(ligand_coords, dtype=torch.float32)
        ligand_atom_types = torch.tensor(ligand_atom_types, dtype=torch.long).unsqueeze(-1)
        ligand_residues = torch.tensor(ligand_residues, dtype=torch.long).unsqueeze(-1)
        ligand_partial_charges = torch.tensor(ligand_partial_charges, dtype=torch.float32).unsqueeze(-1)
        ligand_is_polar = torch.tensor(ligand_is_polar, dtype=torch.float32).unsqueeze(-1)
        ligand_edge_index = torch.tensor(ligand_edge_index, dtype=torch.long)

        protein_coords = torch.tensor(protein_coords, dtype=torch.float32)
        protein_atom_names = torch.tensor(protein_atom_names, dtype=torch.long).unsqueeze(-1)
        protein_residue_names = torch.tensor(protein_residue_names, dtype=torch.long).unsqueeze(-1)
        protein_is_polar = torch.tensor(protein_is_polar, dtype=torch.float32).unsqueeze(-1)
        protein_edge_index = torch.tensor(protein_edge_index, dtype=torch.long)

        ligand_node_features = torch.cat(
            [ligand_atom_types, ligand_coords, ligand_residues, ligand_partial_charges, ligand_is_polar], dim=-1)

        protein_node_features = torch.cat(
            [torch.zeros((protein_atom_names.size(0), 1), dtype=torch.float32),
             protein_atom_names, protein_coords, protein_residue_names, protein_is_polar], dim=-1)

        ligand_data = Data(
            x=ligand_node_features,
            edge_index=ligand_edge_index,
        )

        protein_data = Data(
            x=protein_node_features,
            edge_index=protein_edge_index
        )

        combined_edge_index = self.merge_edge_indices(ligand_edge_index, protein_edge_index)
        combined_node_features = torch.cat([ligand_data.x, protein_data.x], dim=0)

        combined_data = Data(
            interaction_name=interaction_name,
            x=combined_node_features,
            edge_index=combined_edge_index,
            y=torch.tensor(affinity, dtype=torch.float32)

        )

        return combined_data

    def merge_edge_indices(self, ligand_edge_index, protein_edge_index):
        num_ligand_nodes = ligand_edge_index.max().item() + 1
        protein_edge_index = protein_edge_index + num_ligand_nodes
        combined_edge_index = torch.cat([ligand_edge_index, protein_edge_index], dim=1)
        return combined_edge_index


def create_pyg_dataloader(data_dir, embedding_path, batch_size, shuffle=True):
    dataset = ProteinLigandDataset(data_dir, embedding_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



def compute_metrics(predictions, targets):
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    if len(predictions) > 1:
        spearman_corr, _ = spearmanr(predictions.flatten(), targets.flatten())
    else:
        spearman_corr = 0.0
    return rmse, mae, spearman_corr



def check_data_overlap(train_set_path, val_set_path):
    train_files = set(os.listdir(train_set_path))
    val_files = set(os.listdir(val_set_path))
    overlap = train_files & val_files  # 交集
    if overlap:
        print(f"警告：训练集和验证集有重叠样本！重叠文件数量：{len(overlap)}")
    else:
        print("训练集和验证集无重叠样本。")



def train_and_validate(model, train_loader, val_loader, optimizer, device, num_epochs):
    model.train()
    train_losses = []
    val_losses = []
    best_rmse = float('inf')
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
            optimizer.zero_grad()

            data = batch.to(device)
            target = data.y.view(-1, 1)

            output = model(data.x, data.edge_index, batch.batch)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()


        train_losses.append(epoch_loss / len(train_loader))


        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                data = batch.to(device)
                target = data.y.view(-1, 1)
                output = model(data.x, data.edge_index, batch.batch)

                val_predictions.append(output.cpu().numpy())
                val_targets.append(target.cpu().numpy())

        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)
        val_rmse, val_mae, spearman_corr = compute_metrics(val_predictions, val_targets)
        val_losses.append(val_rmse)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_losses[-1]:.4f}')
        print(f'Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, Spearman Correlation: {spearman_corr}')


        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model_state = model.state_dict()
            best_epoch = epoch + 1

    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    best_model_path = f'./saved_model/PAIGN_TOTAL_best_model_epoch_{best_epoch}_rmse_{best_rmse:.4f}.pth'
    torch.save(best_model_state, best_model_path)

    print(f'\nTraining completed. Best RMSE: {best_rmse:.4f} at epoch {best_epoch}. Model saved to {best_model_path}.')

    return train_losses, val_losses, best_epoch, best_rmse



def plot_training_history(train_losses, val_losses, start_epoch=4):
    plt.figure(figsize=(10, 5))

    epochs_range = range(start_epoch, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses[start_epoch - 1:], label='Train Loss')
    plt.plot(epochs_range, val_losses[start_epoch - 1:], label='Validation RMSE')
    plt.title(f'Training Loss and Validation RMSE (Starting from Epoch {start_epoch})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/RMSE')
    plt.legend()
    plt.grid()


    current_time = datetime.now().strftime("%Y%m%d_%H%M")


    if not os.path.exists('./saved_model'):
        os.makedirs('./saved_model')
    plt.savefig(f'./saved_model/PAIGN_TOTAL_{current_time}.png')  

    plt.show()



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    check_data_overlap(training_set_path, validation_set_path)



    train_loader = create_pyg_dataloader(training_set_path, embedding_path, batch_size=64, shuffle=True)
    val_loader = create_pyg_dataloader(validation_set_path, embedding_path, batch_size=32, shuffle=False)




    node_features_dim = 7
    hidden_channels = 128
    num_classes = 1
    model = PAIGN(node_features_dim, hidden_channels, num_classes).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


    num_epochs =
    train_losses, val_losses, best_epoch, best_rmse = train_and_validate(model, train_loader, val_loader, optimizer,
                                                                         device, num_epochs)


    plot_training_history(train_losses, val_losses)


if __name__ == '__main__':
    main()
