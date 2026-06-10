import os
import numpy as np
import torch
import torch.nn.functional as F
from model_def import ConvNet, preprocess_images

def main():
    data_dir = 'data'
    model_path = os.path.join('models', 'cnn_model.pth')
    
    # Load test data
    x_test_raw = np.load(os.path.join(data_dir, 'test_images.npy'))
    y_test_raw = np.load(os.path.join(data_dir, 'test_labels.npy'))
    
    # Preprocess
    x_test = preprocess_images(x_test_raw)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model = ConvNet(num_classes=10)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        x_test_t = x_test_t.to(device)
        outputs = model(x_test_t)
        probs = F.softmax(outputs, dim=1)
        confidences, preds = probs.max(1)
        
    preds = preds.cpu().numpy()
    confidences = confidences.cpu().numpy()
    y_test = y_test_raw.astype('int64')
    
    # Find true 8 predicted as 3
    misclassified_8_as_3 = []
    for i in range(len(y_test)):
        if y_test[i] == 8 and preds[i] == 3:
            misclassified_8_as_3.append((i, confidences[i]))
            
    print(f"Found {len(misclassified_8_as_3)} instances of true '8' predicted as '3'.")
    for idx, conf in misclassified_8_as_3[:10]:
        print(f"Index: {idx}, Confidence: {conf*100:.2f}%")

if __name__ == '__main__':
    main()
