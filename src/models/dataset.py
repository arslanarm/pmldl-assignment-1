from torch.utils.data import Dataset

class DetoxDataset(Dataset):
    def __init__(self, references, translations):
        self.references = references
        self.translations = translations

    def __getitem__(self, idx):
        assert idx < len(self.references['input_ids'])
        item = {key: val[idx] for key, val in self.references.items()}
        item['decoder_attention_mask'] = self.translations['attention_mask'][idx]
        item['labels'] = self.translations['input_ids'][idx]
        return item

    @property
    def n(self):
        return len(self.references['input_ids'])

    def __len__(self):
        return self.n