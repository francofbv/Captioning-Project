import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import wandb  # Optional: for logging
from tqdm import tqdm

from models.action_recognition_model import ActionRecognitionModel
from utils.preprocess import UCF101Dataset, create_data_loaders

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (frames, targets) in enumerate(progress_bar):
        # Move data to device
        frames = frames.to(device)  # [batch_size, num_frames, channels, height, width]
        targets = targets.to(device)  # [batch_size]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(frames)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.3f}',
            'Acc': f'{accuracy:.2f}%'
        })
        
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validating')
        for batch_idx, (frames, targets) in enumerate(progress_bar):
            frames = frames.to(device)
            targets = targets.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.3f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    return avg_loss, accuracy

def main():
    # Configuration
    config = {
        'num_classes': 101,
        'd_model': 2048,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 8192,
        'dropout': 0.1,
        'batch_size': 32,
        'num_frames': 16,
        'frame_size': 224,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Initialize wandb (optional)
    wandb.init(project='action-recognition', config=config)
    
    # Create model
    model = ActionRecognitionModel(
        num_classes=config['num_classes'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(config['device'])
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        root_dir='path/to/UCF101/videos',
        train_annotation_file='path/to/trainlist01.txt',
        test_annotation_file='path/to/testlist01.txt',
        batch_size=config['batch_size'],
        num_frames=config['num_frames'],
        frame_size=config['frame_size']
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config['num_epochs']):
        print(f'\nEpoch {epoch+1}/{config["num_epochs"]}')
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config['device']
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Save checkpoint if best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / f'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
            print(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Save regular checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)

if __name__ == '__main__':
    main()