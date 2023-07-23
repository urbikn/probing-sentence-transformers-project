import torch
from torch import nn
from tqdm import tqdm

from dataset import ProbingDataset, collate_fn


class ProbingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, device='cpu'):
        super().__init__()
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
                    inputs, labels = batch['embedding'].to(self.device), batch['label'].to(self.device)
                    optimizer.zero_grad()
                    output = self.model(inputs)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                
            # Validation phase
            self.model.eval()
            val_loss = 0
            with tqdm(val_loader, desc=f'Validation epoch {epoch}') as pbar:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(pbar):
                        inputs, labels = batch['embedding'].to(self.device), batch['label'].to(self.device)
                        output = self.model(inputs)
                        loss = criterion(output, labels)
                        val_loss += loss.item()
                        pbar.set_postfix({'val_loss': val_loss / (batch_idx + 1)}) # update progress bar with current validation loss
                    
                val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= patience and patience > 0:
                    print(f'Early stopping after epoch {epoch}.')
                    break


    def evaluate(self):
        pass


if __name__ == '__main__':
    train_dataset = ProbingDataset(
        '.embeddings/pov_questions_fourth.txt',
        '.embeddings/sbert.pov_questions_fourth.pt',
        'train'
    ) 

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = ProbingDataset(
        '.embeddings/pov_questions_fourth.txt',
        '.embeddings/sbert.pov_questions_fourth.pt',
        'val'
    ) 

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=collate_fn
    )

    classifier = ProbingClassifier(384, train_dataset.num_classes(), device='cuda')
    classifier.train(train_loader, val_dataloader, nn.CrossEntropyLoss(), torch.optim.Adam(classifier.parameters()), 100, patience=3)




