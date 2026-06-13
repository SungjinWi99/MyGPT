import torch
from torch.utils.data import Dataset

class KoAlpacaDataset(Dataset):
  def __init__(self, dataset, tokenizer, max_len):
    self.dataset = dataset
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __getitem__(self, index):
    instruction, output, url = self.dataset['train'][index].values()
    prompt_text = f"[사용자]\n{instruction}\n\n[챗봇]\n"
    full_text = prompt_text + output

    enc_prompt = self.tokenizer(prompt_text)
    enc_full = self.tokenizer(full_text, max_length=self.max_len-1, truncation=True)

    input_ids = enc_full['input_ids'] + [self.tokenizer.eos_token_id]
    prompt_len = len(enc_prompt['input_ids'])

    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[:len(input_ids)]

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }
  def __len__(self):
    return len(self.dataset['train'])

def collate_fn(batch, pad_token_id):

  input_ids = [b['input_ids'] for b in batch]
  labels = [b['labels'] for b in batch]
  padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
  )
  padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
  )
  return {
      'input_ids': padded_input_ids,
      'labels': padded_labels.long()
  }
