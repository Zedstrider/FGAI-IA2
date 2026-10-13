# Autoregressive Text Generation using RNN Architectures

## Problem Statement and Objective
The objective of this project is to develop an autoregressive text-generation system by implementing and comparing four distinct neural network architectures: Simple RNN, Bidirectional RNN, LSTM, and GRU. The goal is to evaluate their performance in predicting the next character sequentially under comparable experimental conditions and to analyze the influence of sampling temperature on generated text.

## Dataset Description and Source
* **Dataset Name:** Tiny Shakespeare
* **Source Link:** [Andrej Karpathy's Char-RNN Repository](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
* **Characteristics:** The dataset consists of roughly 1.1 million characters from Shakespeare's works. It has a highly structured, older English vocabulary, tokenized at the character level to form a vocabulary size of 65 unique characters.

## Brief Description of Models Implemented
1. **Simple RNN:** A standard recurrent neural network with 512 hidden units.
2. **Bidirectional RNN:** Utilizes a bidirectional wrapper (256 units per direction) that processes sequences both forward and backward. 
3. **LSTM:** A Long Short-Term Memory network (512 units) designed to combat the vanishing gradient problem using memory gating mechanisms.
4. **GRU:** A Gated Recurrent Unit network (512 units) that streamlines the gating mechanism of an LSTM for faster execution.

## Execution Steps and Required Dependencies
* **Dependencies:** Python 3.x, TensorFlow / Keras, NumPy, urllib.
* **Execution:**
  1. Download the script or open the Jupyter Notebook (`.ipynb`).
  2. Run the environment on a GPU (e.g., Google Colab T4).
  3. Execute the data preprocessing cells to map the vocabulary and construct 100-character input-target sequences.
  4. Run the model initialization and compile cells using the Adam optimizer and Sparse Categorical Cross-Entropy.
  5. Run the temperature sampling generation script.

## Results Obtained and Model Comparison
All models were trained for 5 epochs under identical batch (64) and embedding dimensions (256) to ensure fair evaluation.

| Model | Training Time | Final Loss | Prediction Accuracy | Perplexity |
| :--- | :--- | :--- | :--- | :--- |
| **Simple RNN** | 37.26 s | 1.6194 | 51.75% | 5.0500 |
| **Bidirectional RNN*** | 32.96 s | 0.0228 | 99.36% | 1.0231 |
| **LSTM** | 39.45 s | 1.5336 | 53.91% | 4.6350 |
| **GRU** | 30.76 s | 1.4261 | 56.69% | 4.1624 |

* *Note on Bidirectional RNN:* The near-perfect accuracy (99.36%) is a result of data leakage during teacher-forcing. Because it reads the sequence forward and backward simultaneously, it inherently "looks ahead" at the target character during training, making it mathematically incompatible with strict causal autoregressive generation.

## Sampling-Temperature Analysis
Using the seed sequence `"ROMEO: "`, models were evaluated at temperatures 0.2, 0.7, and 1.2.
* **Temperature 0.2 (Low):** Character probabilities were highly skewed toward the safest predictions. Output was generally coherent but suffered from heavy repetition (e.g., Simple RNN repeated "the bear the...", and Bidirectional RNN entered an infinite loop of "the the the...").
* **Temperature 0.7 (Medium):** Achieved the best balance of diversity and contextual consistency. The LSTM and GRU models successfully mimicked the script formatting (generating character names like `BRUTUS:` and `KING RICHARD II:`) while forming logical pseudo-English sentences.
* **Temperature 1.2 (High):** The distribution flattened, resulting in high diversity but poor coherence. Text dispersed into misspelled phonetic variants and garbled vocabulary across all models.

## Conclusion Identifying the Best-Performing Model
The **GRU (Gated Recurrent Unit)** is the most effective model for this autoregressive dataset. Excluding the fundamentally flawed Bidirectional RNN (due to future-context leakage), the GRU achieved the lowest loss (1.4261), the lowest perplexity (4.1624), and the highest prediction accuracy (56.69%) among all valid unidirectional models. Furthermore, it required the shortest training time (30.76s) and generated the most structurally consistent theatrical prose during qualitative temperature sampling.
