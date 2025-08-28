# GPT-2-124-M-Model
GPT2 124 Million Parameter Model From Scratch


gpt2-from-scratch/
│── config/
│   ├── gpt2_small.json        # Model hyperparameters (n_layers, n_heads, hidden_dim, etc.)
│   ├── tokenizer.json         # Tokenizer config/vocab merges (BPE or custom)
│
│── data/
│   ├── raw/                   # Raw dataset (text dumps, wikitext, etc.)
│   ├── processed/             # Preprocessed tokenized data (npz/pt files)
│   ├── dataloader.py          # Dataset + dataloader utilities
│
│── gpt2/
│   ├── __init__.py
│   ├── layers.py              # Attention, MLP, LayerNorm, etc.
│   ├── model.py               # GPT-2 class (forward, generate, etc.)
│   ├── tokenizer.py           # BPE tokenizer implementation
│   ├── utils.py               # Helper functions (masking, positional encodings, etc.)
│
│── training/
│   ├── train.py               # Main training loop
│   ├── optimizer.py           # AdamW, learning rate scheduler, gradient clipping
│   ├── evaluation.py          # Perplexity, loss tracking
│   ├── checkpoint.py          # Save/load checkpoints
│
│── experiments/
│   ├── exp1_small/            # Logs, configs, and checkpoints for one run
│   ├── exp2_finetune/         # Fine-tuning experiment
│
│── scripts/
│   ├── prepare_data.py        # Preprocess dataset
│   ├── sample.py              # Generate text with trained model
│   ├── profile_model.py       # Params count, FLOPs, etc.
│
│── tests/
│   ├── test_tokenizer.py
│   ├── test_layers.py
│   ├── test_model.py
│
│── requirements.txt
│── README.md
│── run.sh                     # Shell script to launch training with configs

