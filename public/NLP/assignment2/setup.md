# AID 729 Assignment 2 — Setup Instructions

## 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Download the spaCy English model
```bash
python -m spacy download en_core_web_sm
```

---

## Exercise 1: Dependency Parser

### Run the parser
```bash
python q1a_parser.py
```
To change the sentence, edit the bottom of `q1a_parser.py`:
```python
run_parser("your sentence here")
```
Output is printed to the terminal and saved as `parse_output1.csv`, `parse_output2.csv` etc.

### Run the verification suite
```bash
python verify.py
```
Runs 20 test sentences and prints pass/fail status, dependency trees, and arc-level checks to the terminal.

---

## Exercise 2: LSTM and GRU

### 2a — Time Series LSTM
```bash
python q2a_lstm_timeseries.py
```
Prints the model architecture and verifies input/output shapes:
- Input:  `torch.Size([32, 50, 1])`  — batch of 32 sequences, 50 timesteps, 1 feature
- Output: `torch.Size([32, 1])`      — one predicted value per sequence

### 2a — Text Generation LSTM
```bash
python q2a_lstm_textgen.py
```
Prints the model architecture and verifies input/output shapes:
- Input:  `torch.Size([16, 100, 128])` — batch of 16 sequences, 100 characters, vocab size 128
- Output: `torch.Size([16, 100, 128])` — one logit per vocabulary token at every position

### 2c — GRU with explicit gates
```bash
python q2c_gru.py
```
Prints the model architecture, verifies output shapes, cross-checks against
PyTorch's built-in `nn.GRU`, and confirms the gates are functioning:
- `Shapes match: True`             — output shape matches nn.GRU
- `Are gates Broken? => False`     — hidden state changes across timesteps

---

## File Structure
```
.
├── requirements.txt
├── SETUP.pdf
├── SETUP.md
├── q1a_parser.py
├── verify.py
├── parse_output1.csv
├── q2a_lstm_timeseries.py
├── q2a_lstm_textgen.py
└── q2c_gru.py
```
