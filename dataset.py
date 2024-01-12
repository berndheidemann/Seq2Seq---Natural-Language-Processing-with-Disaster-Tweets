import torch
from torch.utils.data import Dataset

class DisasterTweetsDataset(Dataset):
    def __init__(self, df, vocab_size=10000, test=False, sequence_length=100, text_pipeline=None, word_to_idx=None):
        self.df = df
        self.vocab_size = vocab_size
        self.test = test
        self.sequence_length = sequence_length
        self.empty_dummy_for_fixed_length = torch.ones(self.sequence_length, dtype=torch.long)
        self.text_pipeline = text_pipeline
        self.word_to_idx = word_to_idx
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x= self.df.iloc[idx]["text"]
        x = self.text_pipeline(x, word_to_idx=self.word_to_idx)
        x = torch.tensor(x)
        x= torch.cat((x, self.empty_dummy_for_fixed_length))[:self.sequence_length]
        return x