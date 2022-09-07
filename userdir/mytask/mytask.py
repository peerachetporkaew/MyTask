import torch
from fairseq.data import FairseqDataset
from fairseq.tasks import LegacyFairseqTask, register_task

MAX = 20

class TensorDataset(FairseqDataset):

    def __init__(self,n):
        self.data = [ torch.rand((10,)) for i in range(0,n)]

    def __getitem__(self, index):
        item = self.data[index] if self.data is not None else None
        sample = {
            'id' : index,
            'data' : item
        }
        return sample

    def collater(self, samples):
        data_list = torch.cat([sample['data'].unsqueeze(0) for sample in samples])
        id = torch.LongTensor([s["id"] for s in samples])
        minibatch = {'id' : id, 'net_inputs' : data_list}
        #print("Minibatch", minibatch)
        return minibatch

    def __len__(self):
        return len(self.data)

    def num_tokens(self,index):
        return 1

@register_task("task1")
class MyTask(LegacyFairseqTask):
    """Task to finetune RoBERTa for Winograd Schemas."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data", metavar="DIR", help="path to data directory; we load <split>.jsonl"
        )


    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        
        self.datasets[split] = TensorDataset(20)
        
    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.datasets = {}

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None



    @classmethod
    def setup_task(cls, args, **kwargs):

        # load data and label dictionaries
        vocab = None

        return cls(args, vocab)

    