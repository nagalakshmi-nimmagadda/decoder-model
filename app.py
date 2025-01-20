import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import os
import json
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [batch_size, seq_len, d_model]"""
        return x + self.pe[:, :x.size(1), :]

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 1024,
                 n_layers: int = 12,
                 n_heads: int = 16,
                 d_ff: int = 4096,
                 max_seq_length: int = 256,
                 dropout: float = 0.1):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_layer = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.01)

        for layer in self.layers:
            nn.init.normal_(layer.self_attention.in_proj_weight, mean=0.0, std=0.01)
            nn.init.normal_(layer.self_attention.out_proj.weight, mean=0.0, std=0.01)

            for name, param in layer.ff.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param, mean=0.0, std=0.01)
                elif 'bias' in name:
                    nn.init.zeros_(param)

        nn.init.normal_(self.final_layer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x, mask=None):
        # Create causal mask if not provided
        if mask is None:
            seq_length = x.size(1)
            mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
            mask = mask.to(x.device)

        x = self.token_embedding(x)
        x = x.transpose(0, 1)  # Convert to sequence-first format
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Convert back to batch-first

        for layer in self.layers:
            x = layer(x, mask=mask)

        output = self.final_layer(x)
        return output

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = 'cpu'):
        """Load a pretrained model from Hugging Face Hub"""
        try:
            # Download config
            config_file = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(config_file) as f:
                config = json.load(f)
            
            print(f"Loaded config: {config}")  # Debug info
            
            # Create model instance
            model = cls(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_layers=config['n_layers'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                max_seq_length=config['max_seq_length'],
                dropout=0.0  # Set to 0 for inference
            )
            
            # Download and load weights
            weights_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
            state_dict = torch.load(weights_file, map_location=device)
            
            print(f"Model has {len(state_dict)} layers")  # Debug info
            
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode
            
            return model.to(device)
            
        except Exception as e:
            raise Exception(f"Error loading model from {model_id}: {str(e)}")
            
def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.5):
    """
    Generate Shakespeare-style text from a prompt.
    Args:
        prompt (str): The input text to continue from
        max_length (int): Maximum number of tokens to generate
        temperature (float): Controls randomness (higher = more random)
    """
    try:
        if not isinstance(max_length, int):
            max_length = 100
        if not isinstance(temperature, float):
            temperature = 0.5
            
        # Load model and tokenizer
        model_id = "ninagala/shakespeare-model"
        tokenizer_file = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
        model = TransformerDecoder.from_pretrained(model_id)
        tokenizer = Tokenizer.from_file(tokenizer_file)
        
        # Set model to evaluation mode
        model.eval()
        
        # Encode the prompt
        tokens = tokenizer.encode(prompt).ids
        input_ids = torch.tensor(tokens).unsqueeze(0)
        
        # Initialize generated text with prompt
        generated_text = prompt
        
        # Generate new tokens
        with torch.no_grad():
            for _ in range(int(max_length)):
                # Get model output
                outputs = model(input_ids)
                
                # Get next token probabilities
                next_token_logits = outputs[:, -1, :].float()  # Convert to float32
                next_token_logits = next_token_logits / temperature
                
                # Apply softmax to get probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Decode the new token
                new_token = tokenizer.decode([next_token.item()])
                
                # Skip if it's a special token
                if next_token.item() in [tokenizer.token_to_id("[PAD]"), 
                                       tokenizer.token_to_id("[EOS]"), 
                                       tokenizer.token_to_id("[UNK]")]:
                    break
                    
                # Add the new token to generated text
                generated_text += new_token
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Optional: Stop if generated text is too long
                if len(generated_text) > max_length * 4:  # Rough character limit
                    break
        
        return generated_text.strip()
    
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}"
        
# Create Gradio interface with explicit types
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=3, placeholder="Enter your prompt here...", label="Prompt"),
        gr.Slider(20, 200, value=100, step=1, label="Maximum Length"),
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Shakespeare Text Generator",
    description="Generate Shakespeare-style text using a transformer decoder.",
    examples=[
        ["To be, or not to be"],
        ["Friends, Romans, countrymen"],
        ["Now is the winter of our discontent"]
    ]
)

if __name__ == "__main__":
    demo.launch() 
