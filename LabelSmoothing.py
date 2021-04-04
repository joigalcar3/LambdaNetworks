import torch.nn as nn

# Code from https://gist.github.com/suvojit-0x55aa/0afb3eefbb26d33f54e1fb9f94d6b609
class LabelSmoothing(nn.Module):
    """
    Negative log-likelihood loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Function of label smoothing according to the equations in page 7 of the paper: 'Rethinking the Inception
        Architecture for Computer Vision'
        """
        # Computes the logsoftmax of the predictions. For each datapoint, there are 10 numbers, each represents a label
        logprobs = nn.functional.log_softmax(x, dim=-1)

        # Obtain the prediction obtained for the right labels. They should be as close to 1 as possible
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))

        nll_loss = nll_loss.squeeze(1)       # Compute the first term of the equation: the predicted value of the labels
        smooth_loss = -logprobs.mean(dim=-1)  # Compute the second term: average among all label predictions
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss    # Sum both terms weighted with the smoothing
        return loss.mean()    # Compute the batch loss
