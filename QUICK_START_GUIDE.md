# Archangel AI Security Expert - Quick Start Guide 🚀

**Ready to use in 3 minutes!** 

---

## 🔑 Step 1: Get Your Hugging Face Token (Free)

1. **Go to:** https://huggingface.co/settings/tokens
2. **Click:** "New token"
3. **Name:** "Archangel AI"
4. **Type:** Read (free tier works great!)
5. **Copy the token** (starts with `hf_`)

# **Example token:** `hf_[your_token_here]`

---

## ⚡ Step 2: Set Your Token

### Option A: Environment Variable (Recommended)
```bash
export HF_TOKEN="your_token_here"
```

### Option B: Interactive (System will ask)
Just run Archangel and it will prompt for your token.

---

## 🛡️ Step 3: Start Using Real AI Security Analysis!

### Analyze Any Target
```bash
python archangel_lightweight.py analyze google.com
```

**What you'll see:**
```
🛡️ ARCHANGEL LINUX - Lightweight Real AI
🌐 Using Hugging Face Inference API...
🧠 Real AI is analyzing with actual neural networks...

🎯 Target: google.com
📝 Type: web_application
📊 AI Confidence: high
⚠️ Threat Level: medium

🧠 REAL AI REASONING:
This appears to be a web application that requires systematic security analysis...
[AI continues with detailed step-by-step reasoning]

📋 AI RECOMMENDATIONS:
1. Begin with passive reconnaissance to minimize detection risk
2. Use rate limiting to prevent target system overload
3. Monitor for defensive responses during assessment
...
```

### Interactive AI Session
```bash
python archangel_lightweight.py interactive
```

**Available commands:**
- `analyze <target>` - Full AI security analysis
- `chat <message>` - Chat with the AI security expert
- `kernel` - Demo AI-kernel integration
- `quit` - Exit

### Example Interactive Session
```
CloudAI> analyze https://example.com
🎯 REAL AI CLOUD ANALYSIS: https://example.com
🧠 AI is analyzing with real neural networks...
[Detailed AI analysis follows]

CloudAI> chat How do I secure a web application?
🤖 AI: To secure a web application, I recommend following the OWASP Top 10...
[AI provides educational security guidance]

CloudAI> kernel
⚡ REAL AI-KERNEL INTEGRATION
🧠 Real AI analyzing kernel context...
🎯 Real AI Decision: MONITOR
[Shows how AI makes real-time kernel security decisions]
```

---

## 🚀 Advanced Features

### Full Local AI (No Internet Required)
```bash
python archangel_ai.py analyze 192.168.1.100
```
- Downloads and runs AI models locally
- Complete privacy (no data sent to cloud)
- Autonomous tool execution with SmolAgents

### Kernel Integration Demo
```bash
python hybrid_demo.py
```
- Shows complete hybrid architecture
- Kernel-userspace AI communication
- Real-time security decision making

---

## 🎯 What Makes This Special?

### Real AI Reasoning
Unlike other tools, Archangel's AI **actually thinks** about security:

```
🧠 AI THINKING PROCESS:
1. "I see this is a web application on port 443"
2. "HTTPS suggests they care about security"  
3. "I should start with passive reconnaissance"
4. "Let me check for common vulnerabilities systematically"
5. "Based on the tech stack, I'll focus on OWASP Top 10"
```

### Educational Value
The AI **explains** security concepts while working:
- Why it chose specific tools
- What vulnerabilities it's looking for
- How to interpret the results
- Best practices for security testing

### Transparent Decision Making
You can see **exactly why** the AI makes each decision:
- Analysis methodology
- Risk assessment reasoning
- Confidence levels and justification
- Alternative approaches considered

---

## 🛠️ System Modes

### 1. Cloud AI Mode (Recommended for beginners)
- **File:** `archangel_lightweight.py`
- **Requirements:** HF token + internet
- **Benefits:** Fast setup, no downloads, always updated models

### 2. Local AI Mode (Privacy focused)
- **File:** `archangel_ai.py`  
- **Requirements:** 8GB+ RAM, optional HF token
- **Benefits:** Complete privacy, offline operation, autonomous agents

### 3. Hybrid Architecture Mode (Advanced)
- **File:** `hybrid_demo.py`
- **Requirements:** Linux system, kernel headers
- **Benefits:** Kernel integration, real-time decisions, <1ms responses

---

## 💡 Pro Tips

### Token Management
```bash
# Set permanently in your shell profile
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Best Practices
1. **Start with lightweight mode** to get familiar
2. **Use interactive mode** to learn from the AI
3. **Ask the AI to explain** its reasoning
4. **Try different target types** (domains, IPs, web apps)

### Educational Usage
```bash
# Learn about web security
python archangel_lightweight.py interactive
CloudAI> chat Explain SQL injection attacks
CloudAI> analyze vulnerable-web-app.com

# Understand network security  
CloudAI> analyze 192.168.1.1
CloudAI> chat What is port scanning and why is it used?
```

---

## 🔧 Troubleshooting

### Token Issues
```bash
# Check if token is set
echo $HF_TOKEN

# Test token manually
curl -H "Authorization: Bearer $HF_TOKEN" https://api-inference.huggingface.co/models/gpt2
```

### Model Access Issues
- Some models require approval - use public models first
- Rate limiting: wait a few minutes between large requests
- Model loading: first request may be slow as models load

### Connection Issues
```bash
# Test basic connectivity
python -c "import requests; print(requests.get('https://huggingface.co').status_code)"

# Try different model if current one fails
# The system automatically tries multiple fallback models
```

---

## 🎓 Learning Path

### Beginner
1. **Start here:** `python archangel_lightweight.py interactive`
2. **Try:** `analyze google.com`
3. **Learn:** Ask AI to explain its reasoning
4. **Practice:** Analyze different types of targets

### Intermediate  
1. **Advanced analysis:** `python archangel_ai.py analyze target.com`
2. **Autonomous tools:** Choose 'y' when prompted for tool execution
3. **Kernel demo:** `python archangel_ai.py kernel`

### Advanced
1. **Full architecture:** `python hybrid_demo.py`
2. **Build kernel module:** `cd kernel && make`
3. **Extend system:** Modify AI reasoning or add new tools

---

## 🎉 You're Ready!

**Start with this simple command:**
```bash
export HF_TOKEN="your_token_here"
python archangel_lightweight.py analyze example.com
```

**Watch the AI think step-by-step about security!** 🤖🛡️

---

*"The first AI that understands security, not just automates it."*
