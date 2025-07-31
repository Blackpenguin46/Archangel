# M2 MacBook Setup Guide (16GB RAM)

Optimized setup for Apple M2 MacBook with 16GB RAM - focuses on lightweight, efficient models that actually run on your hardware.

## 🍎 M2 MacBook Optimizations

### Hardware Constraints
- **RAM**: 16GB total (need to leave ~4GB for macOS)
- **No discrete GPU**: Using Apple Silicon unified memory
- **Storage**: SSD with good sequential read/write

### Model Selection Strategy
- **Avoid**: 8B+ parameter models (require 16-32GB RAM just for the model)
- **Use**: 117M-345M parameter models (1-2GB RAM usage)
- **Fine-tune**: With LoRA for memory efficiency

## 🚀 Quick Setup for M2 MacBook

### 1. Clone and Setup
```bash
git clone https://github.com/Blackpenguin46/Archangel.git
cd Archangel

# Use M2-optimized configuration
cp config/m2_macbook_config.json config/config.json
```

### 2. Install Lightweight Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install minimal requirements optimized for M2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.30.2 datasets==2.12.0 huggingface_hub==0.15.1
pip install requests numpy pandas aiofiles psutil
pip install pydantic click tqdm rich pyyaml
```

### 3. Test Installation
```bash
# Quick test - should complete in under 30 seconds
python3 -c "
import torch
import transformers
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ Apple Silicon support: {torch.backends.mps.is_available()}')
"
```

## 🧠 M2-Optimized AI Models

### Primary Models (All Lightweight)

**1. DialoGPT-small (117M parameters)**
- Memory usage: ~1GB
- Training time: 10-15 minutes
- Good for conversational AI

**2. DialoGPT-medium (345M parameters)**  
- Memory usage: ~2GB
- Training time: 20-30 minutes
- Better quality responses

**3. DistilGPT2 (82M parameters)**
- Memory usage: ~500MB
- Training time: 5-10 minutes
- Fastest option

### Memory Usage Comparison
```
Model                Parameters    RAM Usage    Training Time
DialoGPT-small      117M          ~1GB         10-15 min
DialoGPT-medium     345M          ~2GB         20-30 min  
DistilGPT2          82M           ~500MB       5-10 min
GPT2                124M          ~1GB         10-15 min

❌ Foundation-Sec-8B  8B           ~16GB        N/A (too large)
❌ Llama-3-8B         8B           ~16GB        N/A (too large)
```

## 🔧 M2-Specific Configuration

### Training Settings
```python
# Optimized for M2 MacBook 16GB
config = {
    "model_name": "microsoft/DialoGPT-small",
    "max_length": 512,        # Reduced from 2048
    "batch_size": 1,          # Small batch size
    "gradient_accumulation_steps": 8,  # Simulate larger batches
    "fp16": True,             # Memory efficiency
    "dataloader_num_workers": 1,       # Single worker
    "low_memory_mode": True
}
```

### Memory Management
```python
# Enable M2 Metal Performance Shaders if available
import torch
if torch.backends.mps.is_available():
    device = "mps"
    print("🚀 Using Apple Silicon GPU acceleration")
else:
    device = "cpu"
    print("💻 Using CPU (still fast on M2)")
```

## 🧪 Test the Training Pipeline

### 1. Quick Test (2-3 minutes)
```bash
python training/deepseek_training_pipeline.py
```

Expected output:
```
🧠 Cybersecurity AI Training Pipeline - Foundation-Sec & Llama Models
✅ Dataset prepared: 54 training examples
✅ Validation set: 6 examples
🚀 Initializing cybersecurity AI training pipeline...
✅ Cybersecurity AI model initialized successfully
🎯 Training cybersecurity AI model on security data...
✅ Training completed!
```

### 2. Monitor Memory Usage
```bash
# In another terminal, monitor memory
watch -n 5 "ps aux | grep python | head -5 && echo '---' && vm_stat"
```

### 3. Expected Performance
- **Model loading**: 10-30 seconds
- **Dataset preparation**: 30-60 seconds  
- **Training (3 epochs)**: 5-15 minutes
- **Total memory usage**: 4-6GB (leaves 10GB for macOS)

## ⚠️ Troubleshooting M2 MacBook Issues

### Memory Issues
```bash
# If you get "killed" or memory errors:
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export TOKENIZERS_PARALLELISM=false

# Reduce batch size even further
# Edit training config: batch_size = 1, max_length = 256
```

### Metal Performance Shaders Issues
```bash
# If MPS gives errors, force CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Or disable MPS entirely  
python -c "
import torch
torch.backends.mps.is_available = lambda: False
"
```

### Slow Training
```bash
# Enable optimizations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Use smaller dataset
python training/deepseek_training_pipeline.py --quick-test
```

## 📊 Performance Expectations

### Realistic M2 MacBook Performance

**Training a 117M model on cybersecurity data:**
- Setup time: 1-2 minutes
- Model download: 2-5 minutes (one-time)
- Training: 10-15 minutes
- Total memory: 4-6GB
- CPU usage: 70-90%

**What you WON'T be able to do:**
- ❌ Run 8B+ parameter models (need 32GB+ RAM)
- ❌ Large batch training (memory limited)
- ❌ Multiple models simultaneously

**What you CAN do:**
- ✅ Train small cybersecurity-focused models
- ✅ Fine-tune with LoRA efficiently  
- ✅ Run autonomous security agents
- ✅ Process security datasets
- ✅ Deploy lightweight AI for security analysis

## 🎯 Optimized Workflow

### 1. Start with Smallest Model
```bash
# Use DialoGPT-small first
python training/deepseek_training_pipeline.py
```

### 2. If Successful, Try Medium
```bash
# Edit config to use DialoGPT-medium
# Then retrain for better quality
```

### 3. Deploy for Security Operations
```bash
# Use trained model in autonomous agents
python archangel_autonomous_system.py --model-path ./trained_models/cybersec_ai
```

## 🚀 Next Steps

Once your lightweight model is trained:

1. **Test Security Agents**: Run autonomous security operations
2. **Add Security Tools**: Integrate with lightweight security analysis
3. **Expand Gradually**: Add more capabilities as needed
4. **Consider Cloud**: For larger models, use cloud training

Remember: **A well-trained 117M model focused on cybersecurity can be surprisingly effective** - it's better to have a working lightweight system than an unusable 8B model that crashes your MacBook!

---

*Optimized for Apple M2 MacBook with 16GB RAM - Practical AI that actually works on your hardware.*