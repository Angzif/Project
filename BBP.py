import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# -------------------------------
# Step 1: Data Generation and Preprocessing
# -------------------------------
def generate_synthetic_data(num_samples=1000, max_items=50, container_size=(100, 100, 100)):
    """
    Generate synthetic training and testing data for the Bin Packing Problem.
    """
    data = []

    for _ in range(num_samples):
        items = []
        num_items = np.random.randint(10, max_items)
        for _ in range(num_items):
            length, width, height = np.random.randint(1, container_size[0] // 2, size=3)
            items.append([length, width, height])
        data.append(items)

    return data

def load_task3_data(file_path):
    """
    Load and preprocess task3 data from a CSV file.
    """
    df = pd.read_csv(file_path)
    data = []

    for _, group in df.groupby('sta_code'):
        items = []
        for _, row in group.iterrows():
            for _ in range(row['qty']):
                items.append([row['长(CM)'], row['宽(CM)'], row['高(CM)']])
        data.append(items)

    return data

# -------------------------------
# Step 2: Dataset and DataLoader
# -------------------------------
class BPPDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items = self.data[idx]
        input_tensor = torch.tensor(items, dtype=torch.float32)
        return input_tensor

def collate_fn(batch):
    """
    Custom collate function to handle variable-sized inputs by padding.
    """
    max_len = max(b.shape[0] for b in batch)
    padded_inputs = torch.zeros(len(batch), max_len, 3)

    for i, b in enumerate(batch):
        padded_inputs[i, :b.shape[0], :] = b

    return padded_inputs

# -------------------------------
# Step 3: Model (Encoder-Decoder)
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, _ = self.lstm(x, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):
        hidden, cell = self.encoder(src)
        batch_size, seq_len, _ = src.size()
        decoder_input = torch.zeros(batch_size, seq_len, hidden.size(2), device=src.device)
        output = self.decoder(decoder_input, hidden, cell)
        return output

# -------------------------------
# Step 4: Training Loop
# -------------------------------
def train(model, dataloader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs in dataloader:
            optimizer.zero_grad()

            outputs = model(inputs)
            targets = inputs.clone()  # Use inputs as dummy targets for this example
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# -------------------------------
# Step 5: Testing and Prediction
# -------------------------------
def predict(model, dataloader, container_sizes, output_file="results.csv"):
    """
    Predict the placement of items in containers and save results to a file.
    """
    model.eval()
    results = []

    with torch.no_grad():
        for inputs in dataloader:
            outputs = model(inputs)
            for batch_idx, output in enumerate(outputs):
                items = inputs[batch_idx]
                placements = []
                for item_idx, item in enumerate(items):
                    container_idx = np.random.choice(len(container_sizes))  # Placeholder logic
                    position = (0, 0, 0)  # Placeholder logic
                    rotation = False  # Placeholder logic
                    placements.append({
                        "item_idx": item_idx + 1,
                        "container": container_idx,
                        "position": position,
                        "rotation": rotation
                    })
                results.append(placements)

    # Save results to a CSV file
    with open(output_file, "w") as f:
        f.write("Order,Item,Container,Position,Rotation\n")
        for order_idx, placements in enumerate(results):
            for placement in placements:
                f.write(f"{order_idx + 1},{placement['item_idx']},{placement['container']},\"{placement['position']}\",{placement['rotation']}\n")

    print(f"Results saved to {output_file}")

# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    # Generate synthetic data for training
    train_data = generate_synthetic_data()
    train_dataset = BPPDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Define model
    input_dim = 3
    hidden_dim = 128
    output_dim = 3
    encoder = Encoder(input_dim, hidden_dim)
    decoder = Decoder(hidden_dim, output_dim)
    model = Seq2Seq(encoder, decoder)

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_epochs = 20

    # Train the model
    train(model, train_loader, optimizer, criterion, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "bpp_model.pth")

    # Load task3 data
    test_data = load_task3_data('task3.csv')
    test_dataset = BPPDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Define container sizes
    container_sizes = [(35, 23, 13), (37, 26, 13), (38, 26, 13),
                       (40, 28, 16), (42, 30, 18), (42, 30, 40),
                       (52, 40, 17), (54, 45, 36)]

    # Predict placement
    predict(model, test_loader, container_sizes, output_file="results.csv")
