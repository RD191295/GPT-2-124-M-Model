<!-- Banner (replace with your own image if available) -->
<p align="center">
  <img src="https://raw.githubusercontent.com/RD191295/GPT-2-124-M-Model/main/assets/banner.png" alt="GPT-2 Banner" width="80%">
</p>

<h1 align="center">🤖 GPT-2 124M — From Scratch</h1>

<p align="center">
  <b>A clean PyTorch implementation of GPT-2 (124M parameters)</b><br/>
</p>

<p align="center">
  <a href="https://github.com/RD191295/GPT-2-124-M-Model/stargazers">
    <img src="https://img.shields.io/github/stars/RD191295/GPT-2-124-M-Model?style=social" alt="Stars"/>
  </a>
  <a href="https://github.com/RD191295/GPT-2-124-M-Model/issues">
    <img src="https://img.shields.io/github/issues/RD191295/GPT-2-124-M-Model" alt="Issues"/>
  </a>
  <a href="https://github.com/RD191295/GPT-2-124-M-Model/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/RD191295/GPT-2-124-M-Model" alt="License"/>
  </a>
</p>

---

## ✨ Features

- ⚡ **Faithful GPT-2 Architecture** — Transformer blocks, causal self-attention, GELU feed-forward
- 📚 **Educational** — detailed NumPy-style docstrings for every module
- 🔧 **Modular Design** — easy to customize layers, heads, embedding size
- 🧩 **Pre-Norm Transformer** — stable training with layer norm before sublayers
- 💾 **Checkpoint-Friendly** — model is structured for saving/loading easily

---

## 🏗️ Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/RD191295/GPT-2-124-M-Model/main/assets/architecture.png" alt="GPT2 Architecture" width="70%">
</p>

The model consists of:
- Token + positional embeddings  
- A stack of Transformer blocks  
- Final layer normalization  
- Linear output head → vocabulary logits  

---

## 🚀 Quick Start

### 1️⃣ Install

```bash
git clone https://github.com/RD191295/GPT-2-124-M-Model.git
cd GPT-2-124-M-Model
pip install -r requirements.txt
```

### 2️⃣ Use the Model

```python 
import torch
from model import GPT2Model

model = GPT2Model(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    dropout=0.1,
    qkv_bias=True
)

# dummy input
input_ids = torch.randint(0, 50257, (2, 128))
logits = model(input_ids)
print(logits.shape)  # (2, 128, 50257)
```

---

## 📂 Project Structure

```bash
├── model.py         # TransformerBlock & GPT2Model
├── trainer.py       # (WIP) Training loop
├── utils.py         # (optional) helper functions
├── assets/          # images for README
├── requirements.txt # dependencies
└── README.md
```


## 🙏 Acknowledgments

- [OpenAI GPT-2 (2019)](https://openai.com/research/language-unsupervised)  
- [Andrej Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT)  
- [Build LLM From Scratch-Vizura](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu)

---

## 📖 References

- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019).  
  *Language Models are Unsupervised Multitask Learners*.  
  [📄 Read the paper (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  

---

## 📜 License
This project is licensed under the MIT License
