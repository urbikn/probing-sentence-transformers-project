import torch
from torch import nn
from tqdm import tqdm

from dataset import ProbingDataset, collate_fn


class ProbingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0, device='cpu'):
        super().__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, num_classes),
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

    def train(self, train_loader, val_loader, criterion, optimizer, epochs, patience=0):
        """
        Trains the probing classifier.

        Parameter 'patience' stops training early if the validation loss does not improve for a given number of epochs.
        When patience is set to 0, training will not stop early.
        """
        best_loss = float('inf')
        early_stop_count = 0
        
        for epoch in range(1, epochs + 1):
            # Training phase
            self.model.train()

            with tqdm(train_loader, desc=f'Training epoch {epoch}') as pbar:
                for batch in pbar:
                    inputs, targets = batch['embedding'].to(self.device), batch['label'].to(self.device)
                    optimizer.zero_grad()
                    output = self.model(inputs)
                    loss = criterion(output, targets)
                    loss.backward()
                    optimizer.step()
                
            # Validation phase
            self.model.eval()
            val_loss = 0
            with tqdm(val_loader, desc=f'Validation epoch {epoch}') as pbar:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(pbar):
                        inputs, targets = batch['embedding'].to(self.device), batch['label'].to(self.device)
                        output = self.model(inputs)
                        loss = criterion(output, targets)
                        val_loss += loss.item()
                        pbar.set_postfix({'val_loss': val_loss / (batch_idx + 1)}) # update progress bar with current validation loss
                    
                val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_count = 0
                # Save model
                torch.save(self.model.state_dict(), '.best_probing_classifier.pt')
            else:
                early_stop_count += 1
                if early_stop_count >= patience and patience > 0:
                    print(f'Early stopping after epoch {epoch}.')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('.best_probing_classifier.pt'))

    def evaluate(self, data_loader):
        self.model.eval()
        total_correct = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch['embedding'].to(self.device), batch['label'].to(self.device)
                output = self.model(inputs)

                predictions = nn.functional.softmax(output, dim=1).argmax(1)
                total_correct += (predictions == targets).sum().item()

        accuracy = total_correct / len(data_loader.dataset)
        return accuracy


if __name__ == '__main__':
    data_dir = '.embeddings/pov_questions_fourth.txt'
    embeddings_file = '.embeddings/sbert.pov_questions_fourth.pt'
    batch_size = 32

    # Define train, val, test datasets and dataloaders
    splits = ['train', 'val', 'test']
    datasets = {split: ProbingDataset(data_dir, embeddings_file, split) for split in splits}
    dataloaders = {split: torch.utils.data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True, collate_fn=collate_fn) for split in splits}

    # Train classifier on train and val sets, then evaluate on test set
    classifier = ProbingClassifier(384, datasets['train'].num_classes(), dropout=0.1, device='cuda')
    classifier.train(dataloaders['train'], dataloaders['val'], nn.CrossEntropyLoss(), torch.optim.Adam(classifier.parameters()), 100, patience=10)
    accuracy = classifier.evaluate(dataloaders['test'])

    print(f'Test accuracy: {accuracy:.4f}')