#  LLM Assignment – Transformer Models (Encoder & Decoder)

##  Repository Overview
This repository contains implementations of Transformer-based Language Models as part of the LLM Assignment.  
The project is divided into two experiments, each focusing on a core Transformer architecture used in modern Large Language Models.

---

##  Experiments Included

###  Experiment 1: Transformer Encoder – Autoencoding (Masked Language Model)
📁 Folder: transformer-encoder-autoencoding

Focus Areas:
- Transformer Encoder
- Self-Attention mechanism
- Masked Language Modeling (MLM)
- Autoencoding
- Attention visualization

Example:
Input  : Transformers use [MASK] attention  
Output : Transformers use self attention

---

###  Experiment 2: Transformer Decoder – Autoregression / Seq2Seq
📁 Folder: transformer-seq2seq

Focus Areas:
- Transformer Decoder
- Autoregression
- Causal Masking
- Encoder–Decoder (Seq2Seq)
- Token-by-token text generation

Example:
Input  : AI improves healthcare  
Output : AI enhances medical diagnosis and treatment

---

## 📂 Repository Structure

LLM_Assignment/  
├── transformer-encoder-autoencoding/  
│   ├── dataset.py  
│   ├── attention.py  
│   ├── encoder.py  
│   ├── train_mlm.py  
│   ├── visualize_attention.ipynb  
│   ├── README.md  
│   └── results/  
│  
├── transformer-seq2seq/  
│   ├── dataset.py  
│   ├── encoder.py  
│   ├── decoder.py  
│   ├── transformer.py  
│   ├── attention_masks.py  
│   ├── train.py  
│   ├── inference.py  
│   ├── README.md  
│   └── results/  
│  
└── README.md  

---

##  Technologies Used
- Python 3
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook
- Git and GitHub

---

##  How to Run the Experiments

### Experiment 1
cd transformer-encoder-autoencoding  
python train_mlm.py  
jupyter notebook  

### Experiment 2
cd transformer-seq2seq  
python train.py  
python inference.py  

---

##  Learning Outcomes
- Understanding Transformer Encoder and Decoder architectures
- Difference between self-attention and cross-attention
- Masked Language Modeling
- Autoregressive text generation
- Causal masking in Transformers
- Seq2Seq learning using Encoder–Decoder models

---

##  Conceptual Comparison

Feature | Encoder (Exp 1) | Encoder–Decoder (Exp 2)
--------|----------------|------------------------
Masked Prediction | Yes | No
Autoregression | No | Yes
Seq2Seq Tasks | No | Yes
Cross-Attention | No | Yes
Text Generation | Limited | Full

---

##  Conclusion
This repository demonstrates the fundamental working principles of modern Large Language Models by implementing both encoder-only and encoder–decoder Transformer architectures.  
The experiments provide practical insight into how models such as BERT and GPT operate internally.

---

##  Author
Name: Naveenkumar N 
Course: BE Computer Science and Engineering  
Institution: MIT Chennai  

---

##  Status
✔ Experiment 1 Completed  
✔ Experiment 2 Completed  
✔ Code Verified  
✔ Outputs Generated  
✔ GitHub Submission Ready  
