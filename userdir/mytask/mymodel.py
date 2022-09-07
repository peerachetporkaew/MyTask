import torch
import torch.nn as nn

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)

@register_model("mymodel")
class FairseqFFNModel(BaseFairseqModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument("--hidden-dim", type=int, metavar="N",help="hidden dim", default=10)
   
    def __init__(self, args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(args.hidden_dim,1)
        
    def forward(self, netinput, **kwargs):
        out = self.linear(netinput)
        return out

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        mymodel_base(args)
        return cls(args)


@register_model_architecture("mymodel","base")
def mymodel_base(args):
    args.hidden_dim = getattr(args,"hidden_dim",10)