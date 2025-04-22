from data import PGSDataset, get_dataloader

dataset = PGSDataset()
sample = dataset[0]

print(type(sample['input_ids']))  # Now shows <class 'torch.Tensor'>
print(sample['input_ids'].shape)   # Shows torch.Size([sequence_length])

loader = get_dataloader(batch_size=2)
batch = next(iter(loader))
print(batch['input_ids'].shape)    # Shows torch.Size([2, max_length])