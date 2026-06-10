import io
import os
import base64
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, render_template_string, jsonify, request
from model_def import ConvNet, preprocess_images

app = Flask(__name__)

model_path = os.path.join('models', 'cnn_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model and test data
model = None
x_test = None
y_test = None
model_meta = {}

def load_model_and_data():
    global model, x_test, y_test, model_meta
    
    # Load test data
    data_dir = 'data'
    try:
        x_test_path = os.path.join(data_dir, 'test_images.npy')
        y_test_path = os.path.join(data_dir, 'test_labels.npy')
        if os.path.exists(x_test_path) and os.path.exists(y_test_path):
            x_test = np.load(x_test_path)
            y_test = np.load(y_test_path)
            print("Test data loaded successfully.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        
    # Load model
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model = ConvNet(num_classes=10)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()
            model_meta = {
                'test_acc': checkpoint.get('test_acc', 0.98),
                'epochs': checkpoint.get('hyperparameters', {}).get('epochs', 5),
                'batch_size': checkpoint.get('hyperparameters', {}).get('batch_size', 64),
                'subset': checkpoint.get('hyperparameters', {}).get('subset', 20000)
            }
            print("PyTorch CNN model loaded successfully.")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")

def preprocess_canvas_image(img_base64):
    if ',' in img_base64:
        img_base64 = img_base64.split(',')[1]
    
    img_data = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_data)).convert('L') # Convert to Grayscale
    
    np_img = np.array(img)
    non_zero = np.argwhere(np_img > 10)
    
    if len(non_zero) > 0:
        # Get bounding box of the non-zero pixels
        min_y, min_x = non_zero.min(axis=0)
        max_y, max_x = non_zero.max(axis=0)
        
        # Crop the drawing
        cropped = img.crop((min_x, min_y, max_x + 1, max_y + 1))
        
        # Make it square with padding to match MNIST centering
        width, height = cropped.size
        max_dim = max(width, height)
        padded_size = int(max_dim * 1.4)
        padded_img = Image.new('L', (padded_size, padded_size), 0)
        
        paste_x = (padded_size - width) // 2
        paste_y = (padded_size - height) // 2
        padded_img.paste(cropped, (paste_x, paste_y))
        
        # Resize to 28x28
        img = padded_img.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1, 1, 28, 28)
    return arr

def to_base64_png(arr):
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 1:
        side = int(np.sqrt(len(arr)))
        arr = arr[:side*side].reshape(side, side)
    
    if arr.max() <= 1.0:
        arr = (arr * 255).astype('uint8')
    else:
        arr = arr.astype('uint8')
        
    img = Image.fromarray(arr, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Handwritten Character Recognition - CNN Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(20, 26, 46, 0.6);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.4);
            --secondary: #10b981;
            --secondary-glow: rgba(16, 185, 129, 0.3);
            --danger: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
            padding: 40px 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            letter-spacing: -0.025em;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }

        @media (min-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1.1fr 0.9fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .card:hover {
            border-color: rgba(255, 255, 255, 0.12);
            transform: translateY(-2px);
        }

        .card-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 20px;
            color: var(--text-main);
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .card-title span.dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--primary);
            box-shadow: 0 0 10px var(--primary);
        }

        .card-title span.dot.secondary {
            background-color: var(--secondary);
            box-shadow: 0 0 10px var(--secondary);
        }

        /* Canvas Container */
        .canvas-area {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }

        .canvas-wrapper {
            position: relative;
            background: #000000;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 0 0 2px var(--border-color), 0 10px 25px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }

        .canvas-wrapper.drawing {
            box-shadow: 0 0 0 2px var(--primary), 0 0 20px var(--primary-glow);
        }

        canvas {
            display: block;
            cursor: crosshair;
            touch-action: none;
        }

        .canvas-controls {
            display: flex;
            gap: 15px;
            width: 100%;
            justify-content: center;
        }

        .btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            outline: none;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .btn-primary {
            background: var(--primary);
            border-color: transparent;
            box-shadow: 0 4px 14px var(--primary-glow);
        }

        .btn-primary:hover {
            background: #4f46e5;
            box-shadow: 0 6px 20px var(--primary-glow);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
        }

        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.2);
            border-color: var(--danger);
        }

        /* Predict Output styles */
        .prediction-layout {
            display: flex;
            flex-direction: column;
            height: 100%;
            justify-content: space-between;
        }

        .prediction-result {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 30px;
            padding: 10px 0 25px 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 25px;
        }

        .pred-digit-box {
            width: 90px;
            height: 90px;
            background: rgba(99, 102, 241, 0.1);
            border: 2px solid var(--primary);
            box-shadow: inset 0 0 15px var(--primary-glow), 0 0 15px var(--primary-glow);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3.5rem;
            font-weight: 800;
            color: #c7d2fe;
        }

        .pred-meta {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .pred-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .pred-confidence {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--secondary);
        }

        .probs-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .prob-row {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .prob-digit {
            font-size: 1rem;
            font-weight: 700;
            color: var(--text-muted);
            width: 15px;
        }

        .prob-bar-container {
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }

        .prob-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, #818cf8 100%);
            border-radius: 4px;
            width: 0%;
            transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .prob-bar.active {
            background: linear-gradient(90deg, var(--secondary) 0%, #34d399 100%);
        }

        .prob-value {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            width: 40px;
            text-align: right;
        }

        /* Test evaluation block */
        .eval-section {
            margin-top: 40px;
        }

        .eval-stats {
            display: flex;
            gap: 30px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }

        .stat-badge {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            padding: 15px 25px;
            border-radius: 16px;
            flex: 1;
            min-width: 180px;
            text-align: center;
        }

        .stat-val {
            font-size: 1.8rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 5px;
        }

        .stat-val.highlight {
            color: var(--secondary);
            text-shadow: 0 0 15px var(--secondary-glow);
        }

        .stat-lbl {
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .samples-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }

        @media (min-width: 600px) {
            .samples-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        @media (min-width: 900px) {
            .samples-grid {
                grid-template-columns: repeat(6, 1fr);
            }
        }

        .sample-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 15px;
            text-align: center;
            transition: all 0.2s ease;
        }

        .sample-item:hover {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.15);
            transform: translateY(-2px);
        }

        .sample-img-box {
            background: #000000;
            width: 80px;
            height: 80px;
            margin: 0 auto 10px auto;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(255,255,255,0.05);
        }

        .sample-img-box img {
            width: 100%;
            height: 100%;
            image-rendering: pixelated;
        }

        .sample-lbl {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .sample-pred {
            font-size: 0.85rem;
            font-weight: 700;
        }

        .sample-pred.correct {
            color: var(--secondary);
        }

        .sample-pred.incorrect {
            color: var(--danger);
        }

        .refresh-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
        }

        /* Error state */
        .alert-box {
            text-align: center;
            padding: 50px;
        }
        
        .alert-box h2 {
            margin-bottom: 15px;
            font-size: 1.8rem;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.1);
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border-left-color: var(--primary);
            animation: spin 1s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
      <header>
        <h1>Handwritten Character Recognition</h1>
        <p>Interactive Convolutional Neural Network (CNN) Visualizer — Task 3</p>
      </header>

      {% if model_missing %}
      <div class="card alert-box">
        <div class="spinner"></div>
        <h2 style="color: var(--primary); display: inline-block; vertical-align: middle;">Model Checkpoint Loading / Training</h2>
        <p style="color: var(--text-muted); margin-top: 15px; margin-bottom: 20px;">
          The CNN model is currently training in the background. Please wait for the script to finish and then refresh this page.
        </p>
        <div style="font-family: monospace; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; border: 1px solid var(--border-color); display: inline-block; color: var(--text-muted);">
          python train_cnn.py --epochs 5 --subset 20000
        </div>
      </div>
      {% else %}
      
      <div class="dashboard-grid">
        <!-- Canvas Card -->
        <div class="card">
            <h2 class="card-title"><span class="dot"></span>Interactive Drawing Canvas</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 20px;">
                Draw a single digit (0-9) inside the black board below. The CNN model will predict it automatically!
            </p>
            
            <div class="canvas-area">
                <div class="canvas-wrapper" id="canvas-wrapper">
                    <canvas id="drawing-canvas"></canvas>
                </div>
                
                <div class="canvas-controls">
                    <button class="btn btn-danger" onclick="clearCanvas()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18m-2 0v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6m3 0V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/></svg>
                        Clear Canvas
                    </button>
                    <button class="btn btn-primary" onclick="predictDrawing()">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
                        Predict Digit
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Predictions Card -->
        <div class="card">
            <h2 class="card-title"><span class="dot secondary"></span>CNN Classification probabilities</h2>
            
            <div class="prediction-layout">
                <div class="prediction-result">
                    <div class="pred-digit-box" id="pred-digit">-</div>
                    <div class="pred-meta">
                        <span class="pred-label">Predicted Digit</span>
                        <span class="pred-confidence" id="pred-conf-val">0% Confidence</span>
                    </div>
                </div>
                
                <div class="probs-list">
                    {% for digit in range(10) %}
                    <div class="prob-row">
                        <span class="prob-digit">{{ digit }}</span>
                        <div class="prob-bar-container">
                            <div class="prob-bar" id="bar-{{ digit }}"></div>
                        </div>
                        <span class="prob-value" id="val-{{ digit }}">0%</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
      </div>

      <!-- Test Evaluator Card -->
      <div class="card eval-section">
        <div class="refresh-row">
            <h2 class="card-title"><span class="dot secondary"></span>Test Set Evaluator</h2>
            <button class="btn" onclick="refreshTestSamples()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
                Refresh Samples
            </button>
        </div>
        
        <div class="eval-stats">
            <div class="stat-badge">
                <div class="stat-val highlight" id="stat-accuracy">{{ accuracy_pct }}%</div>
                <div class="stat-lbl">Test Set Accuracy</div>
            </div>
            <div class="stat-badge">
                <div class="stat-val">PyTorch CNN</div>
                <div class="stat-lbl">Architecture Type</div>
            </div>
            <div class="stat-badge">
                <div class="stat-val" id="stat-size">{{ dataset_size }}</div>
                <div class="stat-lbl">Training Images Subset</div>
            </div>
        </div>

        <h3 style="font-size: 1rem; font-weight: 600; margin-bottom: 20px; color: var(--text-muted); display: flex; align-items: center; gap: 8px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><path d="M9 3v18M15 3v18M3 9h18M3 15h18"/></svg>
            Batch Evaluation Samples (12 Random Test Images)
        </h3>
        
        <div class="samples-grid" id="samples-container">
            <!-- Loaded dynamically via JS -->
        </div>
      </div>
      
      <script>
        // Drawing canvas logic
        const canvas = document.getElementById('drawing-canvas');
        const ctx = canvas.getContext('2d');
        const wrapper = document.getElementById('canvas-wrapper');

        canvas.width = 280;
        canvas.height = 280;

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        let isDrawing = false;
        let lastX = 0;
        let lastY = 0;

        function getCoordinates(e) {
            const rect = canvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: clientX - rect.left,
                y: clientY - rect.top
            };
        }

        function startDrawing(e) {
            isDrawing = true;
            wrapper.classList.add('drawing');
            const coords = getCoordinates(e);
            lastX = coords.x;
            lastY = coords.y;
        }

        function draw(e) {
            if (!isDrawing) return;
            e.preventDefault();
            const coords = getCoordinates(e);
            
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(coords.x, coords.y);
            ctx.stroke();
            
            lastX = coords.x;
            lastY = coords.y;
        }

        function stopDrawing() {
            if (!isDrawing) return;
            isDrawing = false;
            wrapper.classList.remove('drawing');
            predictDrawing();
        }

        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseleave', stopDrawing);

        canvas.addEventListener('touchstart', startDrawing);
        canvas.addEventListener('touchmove', draw);
        canvas.addEventListener('touchend', stopDrawing);

        function clearCanvas() {
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            resetPredictionUI();
        }

        function resetPredictionUI() {
            document.getElementById('pred-digit').innerText = '-';
            document.getElementById('pred-conf-val').innerText = '0% Confidence';
            for (let i = 0; i < 10; i++) {
                const bar = document.getElementById(`bar-${i}`);
                const val = document.getElementById(`val-${i}`);
                bar.style.width = '0%';
                bar.classList.remove('active');
                val.innerText = '0%';
            }
        }

        function predictDrawing() {
            const dataUrl = canvas.toDataURL('image/png');
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: dataUrl })
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    console.error(data.error);
                    return;
                }
                
                document.getElementById('pred-digit').innerText = data.prediction;
                document.getElementById('pred-conf-val').innerText = data.confidence + '% Confidence';
                
                data.probabilities.forEach((prob, index) => {
                    const pct = Math.round(prob * 100);
                    const bar = document.getElementById(`bar-${index}`);
                    const val = document.getElementById(`val-${index}`);
                    
                    bar.style.width = pct + '%';
                    val.innerText = pct + '%';
                    
                    if (index === data.prediction) {
                        bar.classList.add('active');
                    } else {
                        bar.classList.remove('active');
                    }
                });
            })
            .catch(err => console.error("Error predicting:", err));
        }

        function refreshTestSamples() {
            const container = document.getElementById('samples-container');
            container.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 40px;"><div class="spinner"></div>Loading test samples...</div>';
            
            fetch('/random_samples')
                .then(res => res.json())
                .then(data => {
                    container.innerHTML = '';
                    data.samples.forEach(item => {
                        const isCorrect = item.true_label === item.pred_label;
                        const predClass = isCorrect ? 'correct' : 'incorrect';
                        
                        const card = document.createElement('div');
                        card.className = 'sample-item';
                        card.innerHTML = `
                            <div class="sample-img-box">
                                <img src="data:image/png;base64,${item.image}" alt="Test image" />
                            </div>
                            <div class="sample-lbl">True Label: <strong>${item.true_label}</strong></div>
                            <div class="sample-pred ${predClass}">
                                Pred: <strong>${item.pred_label}</strong> (${item.confidence}%)
                            </div>
                        `;
                        container.appendChild(card);
                    });
                })
                .catch(err => {
                    console.error("Error fetching samples:", err);
                    container.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--danger); padding: 20px;">Error loading samples</div>';
                });
        }

        document.addEventListener('DOMContentLoaded', () => {
            refreshTestSamples();
        });
      </script>
      {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    global model
    if model is None:
        load_model_and_data()
        
    if model is None:
        return render_template_string(TEMPLATE, model_missing=True)
        
    accuracy_pct = int(round(model_meta.get('test_acc', 0.98) * 100))
    dataset_size = model_meta.get('subset', 20000)
    
    return render_template_string(
        TEMPLATE,
        model_missing=False,
        accuracy_pct=accuracy_pct,
        dataset_size=dataset_size
    )

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        load_model_and_data()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 400
        
    try:
        data = request.json
        img_base64 = data.get('image')
        if not img_base64:
            return jsonify({'error': 'No image data provided'}), 400
            
        arr = preprocess_canvas_image(img_base64)
        arr_t = torch.tensor(arr, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = model(arr_t)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
            confidence = int(round(probs[pred] * 100))
            
        return jsonify({
            'prediction': pred,
            'confidence': confidence,
            'probabilities': [float(p) for p in probs]
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/random_samples')
def random_samples():
    global model, x_test, y_test
    if model is None or x_test is None or y_test is None:
        load_model_and_data()
        if model is None or x_test is None or y_test is None:
            return jsonify({'error': 'Model or data not loaded'}), 400
        
    # Get 12 random indices from the test set
    indices = random.sample(range(len(x_test)), 12)
    samples = []
    
    # Prepare batch
    batch_x_raw = x_test[indices]
    batch_y = y_test[indices]
    
    # Preprocess for model
    batch_x = preprocess_images(batch_x_raw)
    batch_x_t = torch.tensor(batch_x, dtype=torch.float32).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(batch_x_t)
        probs = F.softmax(outputs, dim=1)
        confidences, preds = probs.max(1)
        
    for i in range(12):
        img_b64 = to_base64_png(batch_x_raw[i])
        samples.append({
            'image': img_b64,
            'true_label': int(batch_y[i]),
            'pred_label': int(preds[i].item()),
            'confidence': int(round(confidences[i].item() * 100))
        })
        
    return jsonify({'samples': samples})

if __name__ == '__main__':
    load_model_and_data()
    app.run(host='0.0.0.0', port=5001)
