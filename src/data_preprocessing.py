data_dir = './random_split'

def reader(partition, data_path):
    data = []
    for file_name in os.listdir(os.path.join(data_path, partition)):
        with open(os.path.join(data_path, partition, file_name)) as file:
            data.append(pd.read_csv(file, index_col=None, usecols=["sequence", "family_accession"]))

    all_data = pd.concat(data)

    return all_data["sequence"], all_data["family_accession"]

def build_labels(targets):
    unique_targets = targets.unique()
    fam2label = {target: i for i, target in enumerate(unique_targets, start=1)}
    fam2label['<unk>'] = 0

    print(f"There are {len(fam2label)} labels.")

    return fam2label

def get_amino_acid_frequencies(data):
    aa_counter = Counter()

    for sequence in data:
        aa_counter.update(sequence)

    return pd.DataFrame({'AA': list(aa_counter.keys()), 'Frequency': list(aa_counter.values())})

def build_vocab(data):
    # Build the vocabulary
    voc = set()
    rare_AAs = {'X', 'U', 'B', 'O', 'Z'}
    for sequence in data:
        voc.update(sequence)

    unique_AAs = sorted(voc - rare_AAs)

    # Build the mapping
    word2id = {w: i for i, w in enumerate(unique_AAs, start=2)}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1

    return word2id


class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, word2id, fam2label, max_len, data_path, split):
        self.word2id = word2id
        self.fam2label = fam2label
        self.max_len = max_len

        self.data, self.label = reader(split, data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.preprocess(self.data.iloc[index])
        label = self.fam2label.get(self.label.iloc[index], self.fam2label['<unk>'])

        return {'sequence': seq, 'target' : label}

    def preprocess(self, text):
        seq = []

        # Encode into IDs
        for word in text[:self.max_len]:
            seq.append(self.word2id.get(word, self.word2id['<unk>']))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id['<pad>'] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode
        one_hot_seq = torch.nn.functional.one_hot(seq, num_classes=len(self.word2id), )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1,0)

        return one_hot_seq


if __name__ == "__main__":

    train_data, train_targets = reader("train", data_dir)       # loading train dataset
    print("Peek at few sequences in train dataset:\n", train_data.head())
    fam2label = build_labels(train_targets)     # building labels i.e. y for train dataset
    print("Size of train dataset:", len(train_targets))

    # Data exploration and analysis: a few plots to understand the data at hand

    # Plot for the distribution of family sizes
    # From below plot, one can see that the provided dataset is heavily imbalanced:
    # some families have up to roughly 3k samples while others only have around ten samples.

    f, ax = plt.subplots(figsize=(8, 5))

    sorted_targets = train_targets.groupby(train_targets).size().sort_values(ascending=False)

    sns.histplot(sorted_targets.values, kde=True, log_scale=True, ax=ax)

    plt.title("Distribution of family sizes for the 'train' split")
    plt.xlabel("Family size (log scale)")
    plt.ylabel("# Families")
    plt.show()

    # Plot for the distribution of sequences' lengths
    # One can observe that the sequences' lengths are quite different over the dataset.

    f, ax = plt.subplots(figsize=(8, 5))

    sequence_lengths = train_data.str.len()
    median = sequence_lengths.median()
    mean = sequence_lengths.mean()

    sns.histplot(sequence_lengths.values, kde=True, log_scale=True, bins=60, ax=ax)

    ax.axvline(mean, color='r', linestyle='-', label=f"Mean = {mean:.1f}")
    ax.axvline(median, color='g', linestyle='-', label=f"Median = {median:.1f}")

    plt.title("Distribution of sequence lengths")
    plt.xlabel("Sequence' length (log scale)")
    plt.ylabel("# Sequences")
    plt.legend(loc="best")
    plt.show()

    # Plot for the distribution of AA frequencies
    # Eventually, one can observe that some amino acids (X, U, B, O, Z) are quite rare compared to other ones.
    # We choose to consider those rare amino acids as unknown (<unk>) amino acids but this can be changed.

    f, ax = plt.subplots(figsize=(8, 5))

    amino_acid_counter = get_amino_acid_frequencies(train_data)

    sns.barplot(x='AA', y='Frequency', data=amino_acid_counter.sort_values(by=['Frequency'], ascending=False), ax=ax)

    plt.title("Distribution of AAs' frequencies in the 'train' split")
    plt.xlabel("Amino acid codes")
    plt.ylabel("Frequency (log scale)")
    plt.yscale("log")
    plt.show()

    word2id = build_vocab(train_data)       # building word to index mapping from the training data
    print(f"AA dictionary formed. The length of dictionary is: {len(word2id)}.")

    # Building train, validation, and test dataset
    seq_max_len = 120
    train_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "train")
    dev_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "dev")
    test_dataset = SequenceDataset(word2id, fam2label, seq_max_len, data_dir, "test")

    batch_size = 1
    num_workers = 8

    # Building data loaders for each of the 3 datasets
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    dataloaders['dev'] = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    dataloaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )