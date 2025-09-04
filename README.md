<!-- Banner (replace with your own image if available) -->
<p align="center">
  <img src="https://raw.githubusercontent.com/RD191295/GPT-2-124-M-Model/main/assets/banner.png" alt="GPT-2 Banner" width="80%">
</p>

<h1 align="center">🤖 GPT-2 124M — From Scratch</h1>

<p align="center">
  <b>A clean PyTorch implementation of GPT-2 (124M parameters)</b><br/>
  <i>Educational • Modular • Well-Documented</i>
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
