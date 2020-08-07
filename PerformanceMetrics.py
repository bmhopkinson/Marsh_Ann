import numpy as np
import pdb

class PerformanceMetrics:  #this class is only appropriate for prescence/absence right now
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
        tn = np.where(target == 0, corr, False )  #true negatives
        fp = np.where(target == 0, np.logical_not(corr), False)  #false positives
        fn = np.where(target == 1, np.logical_not(corr), False)  # false negatives

        self.loss =+ loss
        self.true_pos  += np.sum(tp)
        self.true_neg  += np.sum(tn)
        self.false_pos += np.sum(fp)
        self.false_neg += np.sum(fn)
        self.positives  += np.sum(target)
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


class PerformanceMetricsPerClass:
    def __init__(self, n_classes):
        self.loss = 0.0
        self.n_classes = n_classes
        self.pred = np.empty((0, n_classes), int)
        self.target    = np.empty((0, n_classes), int)
        self.true_pos  = np.zeros(n_classes, int)
        self.true_neg  = np.zeros(n_classes, int)
        self.false_pos = np.zeros(n_classes, int)
        self.false_neg = np.zeros(n_classes, int)
        self.positives = np.zeros(n_classes, int)
        self.negatives = np.zeros(n_classes, int)
        self.samples   = 0

        #metrics
        self.precision = np.zeros( n_classes , float)
        self.recall    = np.zeros( n_classes , float)
        self.f1 = np.zeros( n_classes , float)
        self.true_pos_rate   = np.zeros( n_classes , float)
        self.true_neg_rate   = np.zeros( n_classes , float)
        self.false_pos_rate  = np.zeros( n_classes , float)
        self.false_neg_rate  = np.zeros( n_classes , float)
        self.loss_per_sample = np.zeros( n_classes , float)

    def accumulate(self, loss, batch_size, pred, target):

        corr = np.equal(target, pred)
        tp = np.where(target == 1, corr, False )  #true positives
        tn = np.where(target == 0, corr, False )  #true negatives
        fp = np.where(target == 0, np.logical_not(corr), False)  #false positives
        fn = np.where(target == 1, np.logical_not(corr), False)  # false negatives

        self.loss += loss
        self.true_pos  += np.sum(tp,axis=0)
        self.true_neg  += np.sum(tn,axis=0)
        self.false_pos += np.sum(fp,axis=0)
        self.false_neg += np.sum(fn,axis=0)
        self.positives += np.sum(target, axis=0)
        self.negatives += np.sum(np.logical_not(target), axis=0)
        self.samples =+ batch_size


        self.precision = np.divide(self.true_pos, (self.true_pos + self.false_pos) )
        self.recall    = np.divide(self.true_pos, (self.true_pos + self.false_neg) )
        self.f1 = 2 * np.divide( np.multiply(self.precision , self.recall) , (self.precision + self.recall)  )
        self.true_pos_rate  = np.divide( self.true_pos  , self.positives)
        self.true_neg_rate  = np.divide( self.true_neg  , self.negatives)
        self.false_pos_rate = np.divide( self.false_pos , self.negatives)
        self.false_neg_rate = np.divide( self.false_neg , self.positives)
        self.loss_per_sample = np.divide( self.loss , self.samples)
    #    pdb.set_trace()
