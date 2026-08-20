import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn
import numpy as np

# Converts chess position to a 12x8x8 tensor representation
def board_to_tensor(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        channel = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
        tensor[channel, row, col] = 1.0
    return tensor

# Custom PyTorch Dataset that loads from /home/mas/chess-ml/data/*.pgn
class ChessDataset(Dataset):
    def __init__(self, data_dir, max_positions=100000):
        self.X, self.y = [], []
        pgn_files = sorted(glob.glob(os.path.join(data_dir, "*.pgn")))
        print(f"Found {len(pgn_files)} files. Parsing positions...")
        
        for filepath in pgn_files:
            if len(self.X) >= max_positions:
                break
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                while len(self.X) < max_positions:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    res = game.headers.get("Result", "*")
                    target = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else (0.0 if res == "1/2-1/2" else None))
                    if target is None:
                        continue
                    
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        self.X.append(board_to_tensor(board))
                        self.y.append(target)
                        if len(self.X) >= max_positions:
                            break
                            
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        print(f"Loaded {len(self.y)} position matrices.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# Value Network Architecture
class ChessValueNet(nn.Module):
    def __init__(self):
        super(ChessValueNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(-1)

# Training Loop Execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    dataset = ChessDataset(data_dir="/home/mas/chess-ml/data", max_positions=50000)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = ChessValueNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()
        for step, (x_batch, y_batch) in enumerate(loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/3] Step [{step+1}/{len(loader)}] Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "chess_model.pth")
    print("Training complete! Model saved to chess_model.pth.")

if __name__ == "__main__":
    main()
