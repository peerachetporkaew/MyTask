from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch

@register_criterion("mse")
class MSECriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.target_dictionary = None
    
    def forward(self, model, sample):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        #print("Input shape : ",sample['net_inputs'].shape)
        net_output = model(sample['net_inputs'])
        target = sample['net_inputs'].sum(axis=-1).unsqueeze(-1)

        sample_size = len(sample)

        loss = self.compute_loss(net_output, target)
        
        logging_output = {
            "loss": loss.data,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, target):
        return torch.pow(net_output - target,2).sum()

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        
        metrics.log_scalar(
            "loss", loss_sum , 1, round=3
        )
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
        
        