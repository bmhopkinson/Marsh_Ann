import numpy as np

class TrainingMetrics:  #this class is only appropriate for prescence/absence right now
    def __init__(self):
        #raw data - accumulated as batches are processed
        self.loss = 0.0
        self.true_pos = 0
        self.true_neg = 0
        self.false_pos = 0
        self.false_neg = 0
        self.positives = 0
        self.negatives = 0
        self.samples = 0

        #metrics
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.true_pos_rate = 0.0
        self.true_neg_rate = 0.0
        self.false_pos_rate = 0.0
        self.false_neg_rate = 0.0
        self.loss_per_sample = 0.0

    def accumulate(self, loss, batch_size, pred, target):
        corr = np.equal(target, pred)
    	tp = np.where(target == 1, corr, False )  #true positives
        tn = np.where(target_np == 0, corr, False )  #true negatives
    	fp = np.where(target_np == 0, np.logical_not(corr), False)  #false positives
    	fn = np.where(target_np == 1, np.logical_not(corr), False)  # false negatives

    	self.loss =+ loss
    	self.true_pos  += np.sum(tp)
        self.true_neg  += np.sum(tn)
        self.false_pos += np.sum(fp)
        self.false_neg += np.sum(fn)
    	self.postives  += np.sum(target)
        self.negatives += np.sum(np.logical_not(target))
    	self.samples += batch_size

        self.precision = self.true_pos / (self.true_pos + self.false_pos)
        self.recall    = self.true_pos / (self.true_pos + self.false_neg)
        self.f1 = 2*(self.precision * self.recall)/(self.precision + self.recall)
        self.true_pos_rate = self.true_pos / self.positives
        self.true_neg_rate = self.true_neg / self.negatives
        self.false_pos_rate = self.false_pos /self.negatives
        self.false_neg_rate = self.false_neg /self.positives
        self.loss_per_sample = self.loss / self.samples
