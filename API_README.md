# Chinese Legal RAG Text Generation API

A FastAPI-based web service for generating Chinese legal text using trained RNN models (RNN, GRU, LSTM).

## 🚀 Quick Start

### Prerequisites

1. **Trained Models**: Ensure you have trained models in the `outputs_seq_models/` directory
2. **Dataset**: Have your dataset file at `dataset/dataset_clean.csv`
3. **Python**: Python 3.7+ installed

### Easy Startup

#### Windows
```bash
# Double-click or run in command prompt
start_api.bat
```

#### Linux/Mac
```bash
# Make executable and run
chmod +x start_api.sh
./start_api.sh
```

#### Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python run_api.py
```

### With ngrok (for public access)
```bash
# Start with ngrok tunnel
python run_api.py --ngrok --ngrok-token YOUR_NGROK_TOKEN

# Or without auth token (limited sessions)
python run_api.py --ngrok
```

## 📖 API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 🔗 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "lstm",
  "vocab_size": 3547,
  "device": "cpu"
}
```

### 2. Text Generation
```http
POST /generate
```

**Request Body:**
```json
{
  "prompt": "王军的行为是否符合中国刑法关于盗窃罪的构成要件",
  "max_length": 100
}
```

**Response:**
```json
{
  "generated_text": "...generated Chinese legal text...",
  "prompt": "王军的行为是否符合中国刑法关于盗窃罪的构成要件",
  "model_type": "lstm"
}
```

### 3. Model Information
```http
GET /model-info
```

**Response:**
```json
{
  "model_type": "lstm",
  "vocab_size": 3547,
  "device": "cpu",
  "total_parameters": 2891547,
  "trainable_parameters": 2891547,
  "model_architecture": {
    "embedding_dim": 192,
    "hidden_dim": 256,
    "num_layers": 2
  }
}
```

### 4. Test with Sample Prompts
```http
POST /test-prompts
```

Tests the model with predefined Chinese legal prompts and returns an array of generation results.

## 💻 Usage Examples

### Python Client
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Generate text
data = {
    "prompt": "根据《刑法》第二百六十四条，本案适用的量刑幅度是什么",
    "max_length": 150
}
response = requests.post("http://localhost:8000/generate", json=data)
result = response.json()
print(f"Generated: {result['generated_text']}")
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "王军主动认罪并退还赃物，是否应当对量刑产生影响？",
       "max_length": 100
     }'
```

### JavaScript/Node.js
```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Generate text
fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: '你对此案的法律意见或推荐的处理结果是什么',
    max_length: 120
  })
})
.then(response => response.json())
.then(data => console.log(data.generated_text));
```

## ⚙️ Configuration Options

### Command Line Arguments
```bash
python run_api.py --help

Options:
  --host HOST          Host to bind to (default: 0.0.0.0)
  --port PORT          Port to bind to (default: 8000)
  --no-reload          Disable auto-reload
  --ngrok              Start ngrok tunnel
  --ngrok-token TOKEN  ngrok auth token
  --skip-deps          Skip dependency installation
  --skip-checks        Skip file checks
```

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Control GPU usage
- `TORCH_HOME`: Set PyTorch cache directory

## 🌐 ngrok Integration

For public access to your API:

1. **Install ngrok**: Download from https://ngrok.com/download
2. **Get auth token**: Sign up at https://ngrok.com/ 
3. **Run with ngrok**:
   ```bash
   python run_api.py --ngrok --ngrok-token YOUR_TOKEN
   ```

The script will automatically:
- Start the FastAPI server
- Create a public ngrok tunnel
- Display the public URL

## 🔧 Troubleshooting

### Common Issues

1. **Model not found**
   ```
   RuntimeError: No trained models found
   ```
   - Ensure `outputs_seq_models/` directory exists with trained models
   - Check that model files have proper structure: `{model_type}_fold{fold}/`

2. **Dataset not found**
   ```
   RuntimeError: Dataset file not found
   ```
   - Ensure `dataset/dataset_clean.csv` exists
   - Verify CSV has 'text' column

3. **Port already in use**
   ```bash
   # Use different port
   python run_api.py --port 8001
   ```

4. **Memory issues**
   - The model loads into RAM/VRAM on startup
   - For CPU-only: Ensure sufficient RAM
   - For GPU: Check CUDA memory with `nvidia-smi`

### Logs
The API provides detailed logging. Check console output for:
- Model loading progress
- Generation requests
- Error details

## 📊 Performance Notes

- **Startup time**: 10-30 seconds (model loading)
- **Generation speed**: Varies by sequence length and hardware
- **Memory usage**: ~500MB-2GB depending on model and device
- **Concurrent requests**: Supported, but generation is sequential

## 🔒 Security Considerations

For production deployment:
1. Configure CORS properly in `main.py`
2. Add authentication if needed
3. Use HTTPS with reverse proxy
4. Limit request rates
5. Validate input lengths

## 📝 API Schema

The API follows OpenAPI 3.0 specification. Full schema available at `/docs` when server is running.

## 🤝 Contributing

To modify the API:
1. Edit `main.py` for API logic
2. Update `run_api.py` for startup configuration
3. Test with `python run_api.py`
4. Check API docs at `/docs`
