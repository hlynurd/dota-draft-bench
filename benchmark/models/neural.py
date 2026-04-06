"""Baseline 5: Neural network with hero and item embeddings.

Architecture:
  - Hero embedding: hero_id → 32-dim
  - Draft encoder: mean-pool ally embeddings + mean-pool enemy embeddings + buyer embedding → 96-dim
  - Item embedding: item_id → 32-dim
  - Concat draft + item → MLP → P(buy), P(win|buy)
"""

import numpy as np
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark.data import Match, flatten_match
from benchmark.models.base import ItemModel

NUM_HEROES = 200
NUM_ITEMS = 4500  # Valve item IDs go up to ~4206
EMBED_DIM = 32
HIDDEN_DIM = 128
EPOCHS = 10
BATCH_SIZE = 512
LR = 1e-3


class DraftItemNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hero_embed = nn.Embedding(NUM_HEROES, EMBED_DIM, padding_idx=0)
        self.item_embed = nn.Embedding(NUM_ITEMS, EMBED_DIM, padding_idx=0)
        # Input: buyer_embed(32) + ally_pool(32) + enemy_pool(32) + item_embed(32) = 128
        self.mlp = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [P(buy), P(win|buy)]
        )

    def forward(self, buyer, allies, enemies, item):
        """
        buyer: (B,) hero IDs
        allies: (B, 4) hero IDs
        enemies: (B, 5) hero IDs
        item: (B,) item IDs
        """
        buyer_emb = self.hero_embed(buyer)           # (B, 32)
        ally_emb = self.hero_embed(allies).mean(1)   # (B, 32)
        enemy_emb = self.hero_embed(enemies).mean(1) # (B, 32)
        item_emb = self.item_embed(item)             # (B, 32)
        x = torch.cat([buyer_emb, ally_emb, enemy_emb, item_emb], dim=1)
        return self.mlp(x)  # (B, 2)


class NeuralModel(ItemModel):
    def __init__(self, epochs: int = EPOCHS):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for NeuralModel")
        self.epochs = epochs
        self.model: DraftItemNet | None = None
        self.all_items: list[int] = []
        self.hero_buy_rate: dict[int, dict[int, float]] = {}
        self.hero_games: dict[int, int] = defaultdict(int)

    def fit(self, matches: list[Match]) -> "NeuralModel":
        # Build training data: positive (hero bought item) + negative (hero didn't buy item) samples
        buyers, allies_arr, enemies_arr, items_arr = [], [], [], []
        buy_labels, win_labels = [], []

        all_items_set: set[int] = set()
        hero_item_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for match in matches:
            for hero_id, allies, enemies, item_list, won in flatten_match(match):
                self.hero_games[hero_id] += 1
                item_set = set(i for i in item_list if i != 0)
                all_items_set.update(item_set)

                # Pad allies/enemies to fixed size
                a = (allies + [0, 0, 0, 0])[:4]
                e = (enemies + [0, 0, 0, 0, 0])[:5]

                for item_id in item_set:
                    hero_item_counts[hero_id][item_id] += 1
                    # Positive sample
                    buyers.append(hero_id)
                    allies_arr.append(a)
                    enemies_arr.append(e)
                    items_arr.append(item_id)
                    buy_labels.append(1.0)
                    win_labels.append(1.0 if won else 0.0)

                # Negative samples: sample 3 random items the hero didn't buy
                neg_pool = list(all_items_set - item_set)
                if neg_pool:
                    for neg_item in np.random.choice(neg_pool, size=min(3, len(neg_pool)), replace=False):
                        buyers.append(hero_id)
                        allies_arr.append(a)
                        enemies_arr.append(e)
                        items_arr.append(int(neg_item))
                        buy_labels.append(0.0)
                        win_labels.append(0.5)  # don't train win on negatives

        self.all_items = sorted(all_items_set)
        for hero_id, counts in hero_item_counts.items():
            total = self.hero_games[hero_id]
            self.hero_buy_rate[hero_id] = {iid: c / total for iid, c in counts.items()}

        # Convert to tensors
        device = torch.device("cpu")
        t_buyers = torch.tensor(buyers, dtype=torch.long, device=device)
        t_allies = torch.tensor(allies_arr, dtype=torch.long, device=device)
        t_enemies = torch.tensor(enemies_arr, dtype=torch.long, device=device)
        t_items = torch.tensor(items_arr, dtype=torch.long, device=device)
        t_buy = torch.tensor(buy_labels, dtype=torch.float32, device=device)
        t_win = torch.tensor(win_labels, dtype=torch.float32, device=device)

        dataset = TensorDataset(t_buyers, t_allies, t_enemies, t_items, t_buy, t_win)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        self.model = DraftItemNet().to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=LR)
        bce = nn.BCEWithLogitsLoss(reduction="mean")

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            for batch in loader:
                b_buyer, b_allies, b_enemies, b_items, b_buy, b_win = batch
                logits = self.model(b_buyer, b_allies, b_enemies, b_items)

                buy_loss = bce(logits[:, 0], b_buy)
                # Win loss only on positive samples (where buy_label == 1)
                pos_mask = b_buy == 1.0
                if pos_mask.sum() > 0:
                    win_loss = bce(logits[pos_mask, 1], b_win[pos_mask])
                else:
                    win_loss = torch.tensor(0.0)

                loss = buy_loss + win_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"    Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.4f}")

        self.model.eval()
        print(f"  [Neural] Trained on {len(buyers):,} samples, {len(self.all_items)} items")
        return self

    @torch.no_grad()
    def predict_buy(self, hero_id: int, allies: list[int], enemies: list[int]) -> dict[int, float]:
        if not self.model or not self.all_items:
            return {}

        a = (allies + [0, 0, 0, 0])[:4]
        e = (enemies + [0, 0, 0, 0, 0])[:5]
        n = len(self.all_items)

        t_buyer = torch.tensor([hero_id] * n, dtype=torch.long)
        t_allies = torch.tensor([a] * n, dtype=torch.long)
        t_enemies = torch.tensor([e] * n, dtype=torch.long)
        t_items = torch.tensor(self.all_items, dtype=torch.long)

        logits = self.model(t_buyer, t_allies, t_enemies, t_items)
        probs = torch.sigmoid(logits[:, 0]).numpy()

        return {item_id: float(p) for item_id, p in zip(self.all_items, probs)}

    @torch.no_grad()
    def predict_win(self, hero_id: int, item_id: int, allies: list[int], enemies: list[int]) -> float:
        if not self.model:
            return 0.5

        a = (allies + [0, 0, 0, 0])[:4]
        e = (enemies + [0, 0, 0, 0, 0])[:5]

        t_buyer = torch.tensor([hero_id], dtype=torch.long)
        t_allies = torch.tensor([a], dtype=torch.long)
        t_enemies = torch.tensor([e], dtype=torch.long)
        t_items = torch.tensor([item_id], dtype=torch.long)

        logits = self.model(t_buyer, t_allies, t_enemies, t_items)
        return float(torch.sigmoid(logits[0, 1]).item())
