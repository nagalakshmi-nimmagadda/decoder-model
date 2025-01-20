# Shakespeare Text Generator

This project is a Gradio-based web application that generates Shakespeare-style text using a custom TransformerDecoder model. Users can input a prompt, configure parameters like maximum text length and temperature, and generate text interactively.

---

## Features

- **Interactive Text Generation**: Input a prompt and receive text in the style of Shakespeare.
- **Configurable Parameters**:
  - **Maximum Length**: Set the number of tokens to generate.
  - **Temperature**: Adjust the randomness of text generation.
- **Live Preview**: Generate and view text directly in the web interface.
- **Preloaded Transformer Model**: Optimized for fast responses.

---

## Prerequisites

Ensure the following dependencies are installed:

- Python 3.8 or higher
- Gradio
- Transformers
- Torch
- Hugging Face Hub

Install dependencies with:

```bash
pip install gradio transformers torch huggingface_hub

## Model Logs

Epoch 1/10: 100%
 8192/8195 [38:51<00:00,  8.69it/s, loss=0.0908, lr=3.00e-05]

Epoch 1 Summary:
Average Loss: 0.1356
Learning Rate: 3.00e-05
► New best model saved!
Epoch 2/10: 100%
 8192/8195 [20:37<00:00,  8.75it/s, loss=0.0897, lr=6.00e-05]

Epoch 2 Summary:
Average Loss: 0.0913
Learning Rate: 6.00e-05
► New best model saved!

✓ Reached target loss!

## Model Test Results

✓ Found best_model.pt
  • Best loss: 0.0913
  • Saved at epoch: 1

✓ Found tokenizer.json
  • Vocabulary size: 18150

Testing model generation...

✓ Model generation test successful
  • Input: First Citizen
  • Output: First Citizen :

✅ All checks passed! Ready for deployment.
