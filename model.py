import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split fixed-size images into patches and embed each patch as a token."""

    def __init__(
        self,
        image_size: int = 1024,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if image_size <= 0 or patch_size <= 0 or image_size % patch_size != 0:
            raise ValueError("image_size must be positive and divisible by patch_size")

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (batch, channels, height, width)
        expected_shape = (self.in_channels, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_shape:
            raise ValueError(f"Expected images with shape (batch, {expected_shape})")

        patches = self.projection(images)
        return patches.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)


class TransformerBlock(nn.Module):
    """Pre-normalized self-attention and MLP, each with a residual connection."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be positive and divisible by num_heads")
        hidden_dim = int(embed_dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError("mlp_ratio must produce a positive MLP hidden dimension")

        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.attention_dropout(attended)
        return tokens + self.mlp(self.mlp_norm(tokens))


class VisionTransformer(nn.Module):
    """A minimal ViT classifier; inputs are square NCHW image tensors."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth <= 0 or num_classes <= 0:
            raise ValueError("depth and num_classes must be positive")

        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embedding.num_patches + 1, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Replace this head for regression or another downstream task.
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return one image representation per example: (batch, embed_dim)."""
        tokens = self.patch_embedding(images)
        class_tokens = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        tokens = self.norm(self.blocks(tokens))
        return tokens[:, 0]  # Use the class token as the image representation.

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Return raw logits: (batch, num_classes), suitable for CrossEntropyLoss.
        return self.head(self.forward_features(images))


if __name__ == "__main__":
    # Smoke example. Replace random inputs with resized, normalized image batches.
    model = VisionTransformer(num_classes=10)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 224, 224))
    print(f"Logits shape: {tuple(logits.shape)}")
