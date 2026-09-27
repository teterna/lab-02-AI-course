# What to hand in

## 1. Predictions and Measured Results

### Part 0
- Prediction: 16 tokens.
- Measured: 19 tokens.

### Part 1
- Prediction: " Paris" will be the most probable token.
- Measured: " Ast" = 22.87%, " Paris" = 22.08%, Top-10 = 64%.

### Temperature
- Prediction: The top token will remain the same, but its probability will decrease as temperature increases.
- Measured: The top token remained " Ast" at all temperatures.

### Part 3
- Prediction: Kazakh will require more tokens per character than English.
- Measured: English = 0.22, Russian = 1.11, Kazakh = 1.10 tokens/char.

---

## 2. Temperature and Top-p

**Temperature:**  
Temperature changes how concentrated or spread out the probability distribution is while keeping all tokens available.

**Top-p:**  
Top-p removes low-probability tokens and keeps only the smallest set of tokens whose cumulative probability reaches the chosen threshold.

---

## 3. Attention Heads

**Previous-token head:**  
Layer 4, Head 11.

**Most extreme token-0 head:**  
Layer 7, Head 10.

---

## 4. Part 3 Answer

GPT-2 tokenizes Kazakh inefficiently, splitting many words into byte-level fragments. As a result, Kazakh requires significantly more tokens than English and the model often produces broken byte sequences instead of meaningful words.

---

# AI-Use Declaration

ChatGPT was used to explain the laboratory concepts, including tokenization, probability distributions, temperature, top-p sampling, and attention mechanisms, and to help with formatting the written report.

All measurements, token counts, probabilities and conclusions were obtained by running the provided GPT-2 notebook in Google Colab. AI tools were used only for explanation and report preparation.
