import config

class MakeDataset(Dataset):
  def __init__(self, root_dir, split='train', transform=None):
    """
    Args:
      root_dir (string): Directory with all the images.
      split (string): Can be 'train' or 'test'.
      transform (callable, optional): Optional transform to be applied
        on a sample.
    """
    self.transform = transform
    self.files = glob.glob(os.path.join(root_dir, split, '*.jpg'))

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    img = Image.open(self.files[idx])
    img = np.asarray(img)
    if self.transform:
      img = self.transform(img)
    return img
