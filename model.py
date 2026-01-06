import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Convert time-series data into patch embeddings for ViT"""
    def __init__(self, num_features=6, patch_size=8, embed_dim=128):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Linear projection to map patches to embedding space
        self.proj = nn.Linear(patch_size * num_features, embed_dim)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, num_features)
        Output: (batch_size, num_patches, embed_dim)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Ensure sequence length is divisible by patch size
        num_patches = seq_len // self.patch_size
        
        # Reshape into patches
        x = x.view(batch_size, num_patches, self.patch_size, num_features)
        x = x.contiguous().view(batch_size, num_patches, -1)
        
        # Linear projection to embedding dimension
        x = self.proj(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Add positional encoding to embeddings"""
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Learnable positional encoding parameter
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=True)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(context)
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, embed_dim, ff_dim=512, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class ViTEncoderBlock(nn.Module):
    """Single ViT encoder block with attention and feedforward"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(ViTEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class TrendRecognitionFeatureExtractor(nn.Module):
    """ViT-based feature extractor for time-series data"""
    def __init__(self, num_features=6, patch_size=8, embed_dim=128, 
                 num_layers=4, num_heads=8, ff_dim=512, dropout=0.1):
        super(TrendRecognitionFeatureExtractor, self).__init__()
        
        self.patch_embedding = PatchEmbedding(num_features, patch_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        
        self.vit_blocks = nn.ModuleList([
            ViTEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, num_features)
        Output: (batch_size, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through ViT blocks
        for block in self.vit_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block for time-series prediction"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForward(embed_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # Feedforward with residual connection
        ff_out = self.feedforward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class DeepTimeSeriesPredictor(nn.Module):
    """Transformer-based predictor for time-series forecasting"""
    def __init__(self, embed_dim=128, num_layers=4, num_heads=8, 
                 ff_dim=512, dropout=0.1):
        super(DeepTimeSeriesPredictor, self).__init__()
        
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        Output: (batch_size, seq_len, embed_dim)
        """
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.norm(x)
        
        return x


class SPHNet(nn.Module):
    """
    SPH-Net: Stock Price Prediction Hybrid Neural Network
    
    Architecture:
    1. Trend Recognition Feature Extractor (ViT)
    2. Deep Time-Series Predictor (Transformer)
    3. Aggregation layer to combine patch outputs
    4. Output layer for single price prediction
    """
    def __init__(self, num_features=6, patch_size=8, embed_dim=128,
                 vit_num_layers=4, transformer_num_layers=4, num_heads=8,
                 ff_dim=512, dropout=0.1, output_dim=1):
        super(SPHNet, self).__init__()
        
        # Component 1: Trend Recognition Feature Extractor (ViT)
        self.feature_extractor = TrendRecognitionFeatureExtractor(
            num_features=num_features,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=vit_num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Component 2: Deep Time-Series Predictor (Transformer)
        self.time_series_predictor = DeepTimeSeriesPredictor(
            embed_dim=embed_dim,
            num_layers=transformer_num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Aggregation layer: Combine patch outputs into single representation
        # Options: mean pooling, max pooling, or attention-based pooling
        self.aggregation = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Output layer for single price prediction
        self.output_layer = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len, num_features)
        Output: (batch_size, output_dim) - single prediction per sample
        """
        # Stage 1: Extract features using ViT
        # Shape: (batch_size, num_patches, embed_dim)
        features = self.feature_extractor(x)
        
        # Stage 2: Process features with Transformer
        # Shape: (batch_size, num_patches, embed_dim)
        temporal_features = self.time_series_predictor(features)
        
        # Stage 3: Aggregate across patches to get single representation per sample
        # Mean pooling across patch dimension
        aggregated = temporal_features.mean(dim=1)  # Shape: (batch_size, embed_dim)
        
        # Apply aggregation layer
        aggregated = self.aggregation(aggregated)  # Shape: (batch_size, embed_dim)
        
        # Stage 4: Generate single prediction per sample
        predictions = self.output_layer(aggregated)  # Shape: (batch_size, output_dim)
        
        return predictions


# Example usage and model instantiation
def test_sphnet():
    # Model configuration (as per paper)
    config = {
        'num_features': 6,           # Open, High, Low, Close, Adj Close, Volume
        'patch_size': 8,             # Optimal from ablation study
        'embed_dim': 64,            # Patch embedding dimension
        'vit_num_layers': 1,         # ViT layers
        'transformer_num_layers': 1, # Transformer layers
        'num_heads': 2,              # Number of attention heads (optimal)
        'ff_dim': 128,               # Feed-forward dimension
        'dropout': 0.1,              # Dropout rate
        'output_dim': 1              # Single price prediction per sample
    }
    
    # Initialize model
    model = SPHNet(**config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Print model architecture
    print("SPH-Net Model Architecture:")
    print("=" * 80)
    print(model)
    print("=" * 80)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Example forward pass
    batch_size = 256
    seq_len = 64  # Sequence length (must be divisible by patch_size=8)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, config['num_features']).to(device)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {config['output_dim']})")
    
    # Verify shapes match
    assert output.shape == (batch_size, config['output_dim']), \
        f"Output shape mismatch! Got {output.shape}, expected ({batch_size}, {config['output_dim']})"
    
    print("\n✓ Output shape is correct!")
    
    # Example training setup
    print("\n" + "=" * 80)
    print("Example Training Setup:")
    print("=" * 80)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer (as per paper: Adam with lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Loss function: MSE")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"\nModel outputs: {output.shape[0]} predictions per batch")

    print(output)

    print("=" * 80)

if __name__ == "__main__":
    test_sphnet()