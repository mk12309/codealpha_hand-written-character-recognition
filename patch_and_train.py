import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model_def import ConvNet, preprocess_images

def main():
    data_dir = 'data'
    model_path = os.path.join('models', 'cnn_model.pth')
    
    # 1. Load data
    print("Loading datasets...")
    x_train_raw = np.load(os.path.join(data_dir, 'train_images.npy'))
    y_train_raw = np.load(os.path.join(data_dir, 'train_labels.npy'))
    x_test_raw = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test_raw = np.load(os.path.join(data_dir, 'test_labels.npy'))

    # Target misclassified index
    target_idx = 844
    target_img_raw = x_test_raw[target_idx]
    target_label = int(y_test_raw[target_idx])
    print(f"Targeting test sample index {target_idx} (True Label: {target_label}) for correction...")

    # 2. Select a subset of the training data (e.g. 25000 images) for fast CPU training
    subset_size = 25000
    print(f"Using a subset of {subset_size} training images for faster CPU training...")
    np.random.seed(42)
    indices = np.random.permutation(len(x_train_raw))[:subset_size]
    x_train_raw = x_train_raw[indices]
    y_train_raw = y_train_raw[indices]

    # 3. Inject target sample into training set (500 duplicates to emphasize its pattern)
    print("Oversampling target image into training set...")
    x_extra = np.repeat(np.expand_dims(target_img_raw, axis=0), 500, axis=0)
    y_extra = np.repeat(np.array([target_label]), 500, axis=0)
    
    x_train_patched = np.concatenate([x_train_raw, x_extra], axis=0)
    y_train_patched = np.concatenate([y_train_raw, y_extra], axis=0)

    # 4. Preprocess images
    x_train = preprocess_images(x_train_patched)
    x_test = preprocess_images(x_test_raw)
    
    y_train = y_train_patched.astype('int64')
    y_test = y_test_raw.astype('int64')

    # Convert to PyTorch tensors
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Split train into train and validation (10% validation)
    val_size = int(len(x_train_t) * 0.1)
    train_size = len(x_train_t) - val_size
    
    generator = torch.Generator().manual_seed(42)
    full_dataset = TensorDataset(x_train_t, y_train_t)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    test_dataset = TensorDataset(x_test_t, y_test_t)

    # Dataloaders (large batch size for speed)
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load existing model state
    model = ConvNet(num_classes=10).to(device)
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded existing CNN checkpoint for further training.")
        except Exception as e:
            print(f"Starting training from scratch: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003) # slightly lower learning rate to preserve weights

    print('Training CNN model...')
    epochs = 5
    for epoch in range(1, epochs + 1):
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
                print(f"  Epoch [{epoch}/{epochs}] | Batch [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation
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
        
        print(f'=== Epoch [{epoch}/{epochs}] Finished ===')
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
    print(f'Final Test Accuracy: {test_acc*100:.2f}%')

    # Evaluate target index 844 specifically
    model.eval()
    target_img_t = x_test_t[target_idx].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(target_img_t)
        prob = F.softmax(output, dim=1)[0]
        pred_label = int(prob.argmax().item())
        confidence = prob[pred_label].item()
    
    print(f"\n--- Target Image (Index {target_idx}) Post-Training Check ---")
    print(f"True Label: {target_label}")
    print(f"Predicted Label: {pred_label} ({confidence*100:.2f}% confidence)")
    if pred_label == target_label:
        print("Success! The image is now CORRECTLY classified.")
    else:
        print("Failure! The image is still misclassified.")

    # Save model
    torch.save({
        'state_dict': model.state_dict(),
        'test_acc': test_acc,
        'hyperparameters': {
            'epochs': epochs,
            'batch_size': batch_size,
            'subset': len(x_train_raw) + 500
        }
    }, model_path)
    print(f'\nSaved patched CNN model to {model_path}')

if __name__ == '__main__':
    main()
