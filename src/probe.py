import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, Subset, IterableDataset, random_split
from tqdm import tqdm, trange
import numpy as np
import uuid
import os
import pprint
from sklearn import metrics
from dataset import ProbingDataset, collate_fn


class ProbingClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0, device='cpu', ID=None):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        ).to(self.device)

        # Create a .classification folder if it doesn't exist
        if not os.path.exists('.classifiers'):
            os.makedirs('.classifiers')


        # The classification probes ID
        self.ID = ID if ID is not None else str(uuid.uuid4())[:8]


    def forward(self, x):
        output = self.model(x)

        # Flatten output if it has more than 2 dimensions
        if len(output.shape) > 2 and output.shape[1] == 1:
            output = output.squeeze(1)

        return output

    def _train_epoch(self, train_loader, criterion, optimizer, progress_bar=None):
        """
        Performs a single training step (forward and backward pass).

        Returns:
            running_loss (float): The mean loss of the model on the training set.
        """
        running_loss = 0.0

        if progress_bar is None:
            progress_bar = tqdm(train_loader, desc=f'Training epoch')

        for batch in train_loader:
            inputs, targets = batch['embedding'].to(self.device), batch['label'].to(self.device)

            # Forward propagation
            optimizer.zero_grad()
            output = self.forward(inputs)

            # Backward propagation
            breakpoint()
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            # Calculate loss
            running_loss += loss.item()

            # Increment progress bar
            progress_bar.update(1)

        return running_loss / len(train_loader)

    def train(self, train_loader, val_loader, criterion, optimizer, epochs, patience=0, task='no_task'):
        """
        Trains the probing classifier.

        Parameters:
            train_loader (DataLoader): The DataLoader object that provides batches of the training dataset.
            val_loader (DataLoader): The DataLoader object that provides batches of the validation dataset.
            criterion (nn.Module): The criterion (loss function) used to evaluate the performance of the model.
            optimizer (Optimizer): The optimizer used to minimize the criterion.
            epochs (int): The number of times the model will learn from the entire dataset.
            patience (int, optional): The number of epochs to wait for improvement in validation loss before stopping training.
                                    If set to 0, training will not stop early. Default is 0.
            task (str, optional): The task for which the probing classifier is being trained. Default is 'no_task'.

        Returns:
            losses (List[float]): A list of the validation loss at each epoch.
        """
        # Initialize minimum validation loss to a large value
        min_val_loss = np.inf
        # Initialize counter for early stopping
        early_stop_count = 0

        # List to store validation losses
        val_loss_list = []

        with tqdm(range(len(train_loader) * epochs), leave=False) as progress_bar:
            for epoch in range(1, epochs+1):
                progress_bar.set_description(f'Training epoch {epoch}')

                ### Training phase
                self.model.train()
                train_loss = self._train_epoch(train_loader, criterion, optimizer, progress_bar)

                ### Validation phase
                if val_loader is not None:
                    val_loss, val_preds, val_targets = self.evaluate(val_loader, criterion)

                    # Add validation loss to list
                    val_loss_list.append(val_loss)

                    val_loss_avg = val_loss / len(val_loader)
                    accuracy = np.mean(val_preds == val_targets)
                    progress_bar.set_postfix({'Train loss': f'{train_loss:.4f}', 'Val loss': f'{val_loss_avg:.4f}', 'Val accuracy': f'{accuracy:.4f}'})

                    # Check if validation loss has improved
                    if val_loss < min_val_loss:
                        # Update minimum validation loss
                        min_val_loss = val_loss
                        early_stop_count = 0
                        torch.save(self.model.state_dict(), f'.classifiers/best_probing_classifier_{task}_{self.ID}.pt')
                    else:
                        # Increment early stop counter
                        early_stop_count += 1
                        if early_stop_count >= patience and patience > 0:
                            break
                else:
                    progress_bar.set_postfix({'Train loss': f'{train_loss:.4f}'})

        # Load best model
        self.model.load_state_dict(torch.load(f'.classifiers/best_probing_classifier_{task}_{self.ID}.pt'))

        return val_loss_list

    def evaluate(self, test_dataloader, criterion=None):
        """
        This method evaluates the probing classifier on a provided test dataset.

        Parameters:
            test_dataloader (DataLoader): The DataLoader object that provides batches of the test dataset.
            criterion (nn.Module, optional): The criterion (loss function) used to evaluate the performance of the model. 
                                            If not provided, CrossEntropyLoss is used

        Returns:
            loss (float): The evaluation loss over all batches of the test dataset.
            predictions (numpy.ndarray): The model's predictions on the test dataset.
            targets (numpy.ndarray): The actual targets from the test dataset.
        """
        # If no criterion is specified, use CrossEntropyLoss
        if criterion is None:
            criterion = nn.CrossEntropyLoss(reduction='sum')

        total_loss = 0
        all_predictions = None
        all_targets = None

        with tqdm(test_dataloader, desc='Evaluation', leave=False) as progress_bar:
            self.model.eval()
            
            for batch_index, batch in enumerate(progress_bar):
                with torch.no_grad():
                    # Move inputs and targets to the device the model is on
                    inputs = batch['embedding'].to(self.device)
                    targets = batch['label'].to(self.device)

                    # Feed inputs into model and get output
                    output = self.forward(inputs)

                # Compute loss between model output and actual targets
                loss = criterion(output, targets)

                # Update total_loss and num_steps
                total_loss += loss.item()

                # Move output and targets back to cpu for further processing
                output = output.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

                # Determine predictions from output
                predictions = np.argmax(output, axis=1)

                # Add predictions and targets to their respective accumulators
                if all_predictions is None:
                    all_predictions = predictions
                    all_targets = targets
                else:
                    all_predictions = np.append(all_predictions, predictions, axis=0)
                    all_targets = np.append(all_targets, targets, axis=0)
        
        return total_loss, all_predictions, all_targets

    @staticmethod
    def initialize_weights(model):
        """
        Initializes the weights of a module using the Normal distribution with mean 0 and standard deviation 0.01.

        Parameters:
            model (nn.Module): The module to initialize.
        
        Returns:
            model (nn.Module): The re-initialized module.

        """
        # Intialize weights using Kaiming normal initialization
        def init_weights(m):
            if isinstance(model, nn.Linear):
                init.normal_(m.weight.data, mean=0.0, std=0.01)
                m.bias.data.fill_(0.01)
        
        model.apply(init_weights)
        return model

class MDLProbingClassifier():
    """
    Implements the MDL Probing Classifier with Online Coding evaluation.

    Implementation is adapted from Voita and Titov 2020 (https://arxiv.org/pdf/2003.12298.pdf)
    """
    def __init__(self, input_dim, num_classes, device='cpu', ID=None):
        self.portion_ratios = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1.0]
        self.num_classes = num_classes

        model = ProbingClassifier(
            input_dim=input_dim,
            num_classes=self.num_classes,
            dropout=0.3,
            device=device,
            ID=ID
        )
        self.probing_model = model
    

    @staticmethod
    def split_datasets(dataset, fractions, shuffle=True):
        """
        Split a dataset into portions, given by :fractions:

        Parameters:
            dataset (Dataset): The dataset to split.
            fractions (list): A list of fractions to split the dataset into. The fractions should be given as percentages.
            shuffle (bool, optional): Whether to shuffle the dataset before splitting. Default is True.

        Returns:
            train_portions: A list of Subsets, of size len(fractions)
            eval_portions: A list of Subsets, of size len(fractions) - 1
        """
        total_len = len(dataset)
        lengths = [int(frac * total_len) for frac in fractions]

        # Create a shuffled permutation of indices if shuffle is True
        if shuffle:
            # In case that the dataset is not ordered randomly, we need to shuffle it
            shuffled_indices = np.random.permutation(total_len)

            train_portions = [Subset(dataset, shuffled_indices[range(0, length)]) for length in lengths]
            eval_portions = [Subset(dataset, shuffled_indices[range(length, length*2)]) for length in lengths[:-1]]
        else:
            train_portions = [Subset(dataset, range(0, length)) for length in lengths]
            eval_portions = [Subset(dataset, range(length, length*2)) for length in lengths[:-1]]
        
        return train_portions, eval_portions
    
    @staticmethod
    def uniform_code_length(num_classes, train_dataset_size):
        r"""Calculate the uniform code length for a given training task

        Parameters:
            num_classes (int): Number of classes in the probing (classification) task.
            train_dataset_size (int): The size of the full training dataset which the probe was trained on.

        Returns:
            uniform_code_length (float): The uniform code length for the given training/evaluation parameters of
            the probe.
        """
        return train_dataset_size * np.log2(num_classes)

    @staticmethod
    def online_code_length(num_classes, t1, losses):
        r"""Calculate the online code length.

        Parameters:
            num_classes (int): Number of classes in the probing (classification) task.
            t1 (int): The size of the first training block (fraction) dataset.
            losses (List[float]): The list of (test) losses for each evaluation block (fraction)
            dataset, of size len(fractions).

        Returns:
            online_code_length (float): The online code length for the given training/evaluation parameters of
            the probe.
        """
        return t1 * np.log2(num_classes) + sum(losses)


    def analyize(self, train_dataset, val_dataset, test_dataset, collate_fn, batch_size=32, learning_rate=1e-3, train_epochs=50, early_stopping=10, task='no_task'):
        assert self.portion_ratios[-1] == 1.0, 'The last portion ratio must be 1.0'

        # Split the training dataset into incrementally larger subsets based on the portion ratios
        train_dataset_subsets, test_dataset_subsets = self.split_datasets(train_dataset, self.portion_ratios)

        train_dataset_subsets = train_dataset_subsets

        # add the full test dataset to the test dataset subsets
        # since the last portion ratio is 1.0, i.e. the full dataset
        test_dataset_subsets = test_dataset_subsets + [test_dataset]

        # A list to store information about the training of the probes on the online coding subsets
        online_coding_results = []

        # Create progess bar for training the probing classifiers on the subsets
        progress_bar = tqdm(
            enumerate(zip(train_dataset_subsets, test_dataset_subsets)),
            desc=f'Training MDL probe for task {task}',
            total=len(train_dataset_subsets)
        )

        # Train the probing classifier a subset
        for i, (train_subset, test_subset) in progress_bar:
            progress_bar.set_postfix({'Fraction': f'{self.portion_ratios[i]:.4f}'})

            # Reset the model weights
            self.probing_model = ProbingClassifier.initialize_weights(self.probing_model)

            # Create dataloaders for the current subset
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) # Aclual val set
            test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

            criterion = nn.CrossEntropyLoss(reduction='sum')

            # Train the probing classifier on the current subset
            val_loss = self.probing_model.train(
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=torch.optim.Adam(self.probing_model.parameters(), lr=learning_rate),
                epochs=train_epochs,
                patience=early_stopping,
                task=task
            )
            
            # Evaluate the probing classifier on the test dataset
            test_loss, test_preds, test_targets = self.probing_model.evaluate(test_loader, criterion)

            classification_report = metrics.classification_report(test_targets, test_preds, digits=4, zero_division=0, output_dict=True)

            online_coding_results.append({
                'fraction': self.portion_ratios[i],
                'val_loss': sum(val_loss),
                'test_loss': test_loss,
                'classification_report': {
                    'accuracy': classification_report['accuracy'],
                    'macro f1': classification_report['macro avg']['f1-score'],
                    'weighted f1': classification_report['weighted avg']['f1-score'],
                }
            })
        
        breakpoint()
        # Calculate the uniform code length
        num_classes = len(np.unique(test_preds))
        uniform_codelength = self.uniform_code_length(num_classes, len(train_dataset))
        online_codelength = self.online_code_length(
            num_classes=num_classes,
            t1=len(train_dataset_subsets[0]),
            losses=[result['test_loss'] for result in online_coding_results[:-1]] # Except the last full dataset
        )
        
        # Calculate the compression ratio
        compression_ratio = round(uniform_codelength / online_codelength, 2)

        final_report = {
            'uniform_codelength': uniform_codelength,
            'online_codelength': online_codelength,
            'compression_ratio': compression_ratio,
            'online_coding_results': online_coding_results
        }

        return final_report


if __name__ == '__main__':
    # data_dir = '.embeddings/pov_questions_fourth.txt'
    # embeddings_file = '.embeddings/sbert.pov_questions_fourth.pt'
    # embedding_size = 384
    data_dir = '../.embeddings/bigram_shift.txt'
    embeddings_file = '../.embeddings/mpnet-base-v2-layer-12.bigram_shift.pt'
    embedding_size = 768
    batch_size = 32

    # Define train, val, test datasets and dataloaders
    splits = ['train', 'val', 'test']
    datasets = {split: ProbingDataset(data_dir, embeddings_file, split) for split in splits}
    dataloaders = {split: torch.utils.data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True, collate_fn=collate_fn) for split in splits}

    mld_probe = MDLProbingClassifier(embedding_size, datasets['train'].num_classes(), device='cuda', ID='sbert_full_layer')
    report = mld_probe.analyize(datasets['train'], datasets['val'], datasets['test'], collate_fn, task='pov_questions_fourth')

    pprint.pprint(report, indent=4)

    if False:
        mdl = MDLProbingClassifier(384, datasets['train'].num_classes(), device='cuda', ID='sbert_full_layer')
        MDLProbingClassifier.split_datasets(datasets['train'], mdl.portion_ratios)

        # Train classifier on train and val sets, then evaluate on test set
        classifier = ProbingClassifier(384, datasets['train'].num_classes(), dropout=0.3, device='cuda', ID='sbert_full_layer')
        classifier.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            criterion=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(classifier.parameters()),
            epochs=50,
            patience=3
        )
        loss, predictions, targets = classifier.evaluate(dataloaders['test'])

        classification_report = metrics.classification_report(targets, predictions, digits=4, zero_division=0)
        print(classification_report)