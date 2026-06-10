import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_def import ConvNet, preprocess_images

def main(args):
    data_dir = args.data_dir
    # Load numpy arrays
    print("Loading data...")
    try:
        x_train_raw = np.load(os.path.join(data_dir, 'train_images.npy'))
        y_train_raw = np.load(os.path.join(data_dir, 'train_labels.npy'))
        x_test_raw = np.load(os.path.join(data_dir, 'test_images.npy'))
        y_test_raw = np.load(os.path.join(data_dir, 'test_labels.npy'))
    except Exception as e:
        print(f"Error loading numpy arrays: {e}")
        return

    print('Raw shapes:')
    print(f'  Train images: {x_train_raw.shape}, Labels: {y_train_raw.shape}')
    print(f'  Test images:  {x_test_raw.shape}, Labels: {y_test_raw.shape}')

    # Subset data if requested
    if args.subset > 0 and args.subset < len(x_train_raw):
        print(f"Using a subset of {args.subset} training images for faster training.")
        # Shuffle indices to get a representative subset
        np.random.seed(42)
        indices = np.random.permutation(len(x_train_raw))[:args.subset]
        x_train_raw = x_train_raw[indices]
        y_train_raw = y_train_raw[indices]

    # Preprocess
    x_train = preprocess_images(x_train_raw)
    x_test = preprocess_images(x_test_raw)
    
    # Ensure labels are integers
    y_train = y_train_raw.astype('int64')
    y_test = y_test_raw.astype('int64')

    print('Preprocessed shapes (PyTorch format NCHW):')
    print(f'  Train: {x_train.shape}')
    print(f'  Test:  {x_test.shape}')

    # Convert to PyTorch tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Split train into train and validation (10% validation)
    val_size = int(len(x_train_t) * 0.1)
    train_size = len(x_train_t) - val_size
    
    # Fix seed for reproducible splitting
    generator = torch.Generator().manual_seed(42)
    full_dataset = TensorDataset(x_train_t, y_train_t)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    test_dataset = TensorDataset(x_test_t, y_test_t)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize model
    model = ConvNet(num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print('Training CNN model...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
            if (step + 1) % 100 == 0:
                print(f"  Epoch [{epoch}/{args.epochs}] | Batch [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        print(f'=== Epoch [{epoch}/{args.epochs}] Finished ===')
        print(f'    Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}%')
        print(f'    Val Loss:   {epoch_val_loss:.4f}, Val Acc:   {epoch_val_acc*100:.2f}%\n')

    # Test set evaluation
    print("Evaluating model on test set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            test_total += batch_y.size(0)
            test_correct += predicted.eq(batch_y).sum().item()
            
    test_acc = test_correct / test_total
    print(f'Final Test Accuracy: {test_acc*100:.2f}%\n')

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'cnn_model.pth')
    # Save the state dictionary along with model structure information
    torch.save({
        'state_dict': model.state_dict(),
        'test_acc': test_acc,
        'hyperparameters': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'subset': args.subset
        }
    }, model_path)
    print(f'Saved PyTorch CNN model to {model_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data', help='Directory with train_images.npy etc')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--subset', type=int, default=20000, help='Number of training images to use (0 for all)')
    args = parser.parse_args()
    main(args)
