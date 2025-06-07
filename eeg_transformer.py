import torch

def add_noise(x, noise_level=0.01):
    return x + noise_level * torch.randn_like(x)

class AbsPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.pe = torch.nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        self.scale = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.normal_(self.pe, mean=0, std=d_model**-0.5)

    def forward(self, x):
        x = x + (self.scale * self.pe[:, :x.size(1), :])
        return x
    
class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim, heads, max_len, layers):
        super(TransformerEncoder, self).__init__()
        self.pos_enc = AbsPositionalEncoding(dim, max_len)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        x = self.pos_enc(x)                     # Apply positional encoding.
        output = self.transformer_encoder(x)    # Pass through the transformer.
        return output
    
class EEGTransformerModel(torch.nn.Module):
    def __init__(self, channels, class_count, dim, heads, layers, maxlen):
        super(EEGTransformerModel, self).__init__()
        self.input = torch.nn.Linear(channels, dim)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.lstm = torch.nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # You can set this to True if desired
        )
        self.norm2 = torch.nn.LayerNorm(dim)
        self.encoder_stack = TransformerEncoder(dim, heads, maxlen, layers)
        self.classifier = torch.nn.Linear(dim, class_count)

    def forward(self, x):
        x = self.input(x)
        x = self.norm1(x)
        x, _ = self.lstm(x)
        x = self.norm2(x)
        x = self.encoder_stack(x)
        x = self.classifier(x)
        x = x.mean(dim=1)
        return x