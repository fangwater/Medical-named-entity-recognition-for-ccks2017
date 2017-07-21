import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
START_TAG = -2
STOP_TAG = -1


# Helper functions to make the code more readable.
def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CRF(nn.Module):

    def __init__(self, tagset_size):
        super(CRF, self).__init__()
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.tagset_size = tagset_size
        # We add 2 here, because of START_TAG and STOP_TAG
        self.transitions = nn.Parameter(torch.randn(self.tagset_size+2, self.tagset_size+2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        # ADD 2 here because of START_TAG and STOP_TAG
        init_alphas = torch.Tensor(1, self.tagset_size+2).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][ START_TAG ] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        alpha = []
        for feat in feats:
            alphas_t = [] # The forward variables at this timestep
            for next_tag in xrange(self.tagset_size+2):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size+2)
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag)
                # before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
            alpha.append(forward_var)

        terminal_var = forward_var + self.transitions[ STOP_TAG ]
        log_partition_Z = log_sum_exp(terminal_var)
        log_alpha = torch.cat(alpha , 0)
        return log_partition_Z, log_alpha


    def _backward_alg(self, feats):
        # Do the backward algorithm
        # ADD 2 here because of START_TAG and STOP_TAG
        init_betas = torch.Tensor(1, self.tagset_size+2).fill_(0)

        # Wrap in a variable so that we will get automatic backprop
        # This is beta_{T+1} vector
        backward_var = autograd.Variable(init_betas)

        # Iterate through the sentence
        beta = []

        # First calculate beta_{T}, because we do not have Emition_matrix{, T+1}
        # so we need to calculate it seperately
        betas_t = []
        next_tag_var = autograd.Variable(torch.Tensor(1, self.tagset_size+2).fill_(0.))

        for next_tag in xrange(self.tagset_size+2):
            # We add transition score in this way because
            #self.transition[:, next_tag] will not be contiguous, so we cannot use view function
            for i, trans_val in enumerate(self.transitions[:,next_tag]):
                next_tag_var[0,i] = backward_var[0,i]+trans_val
            betas_t.append(log_sum_exp(next_tag_var))
        backward_var = torch.cat(betas_t).view(1, -1)
        beta.append(backward_var)

        # Second we can begin the loop
        # became with Emition_matrix{, T}, beta_{T} to calulate beta_{T-1}
        # slice step has to be greater than 0!!! so urgely in Pytorch
        for j in range(len(feats), 1, -1):
            feat = feats[j-1]
            betas_t = []
            #alphas_t = [] # The forward variables at this timestep
            for next_tag in xrange(self.tagset_size+2):

                emit_score = feat.view(1, -1)

                next_tag_var = backward_var + emit_score
                #
                for i, trans_val in enumerate(self.transitions[:,next_tag]):
                    next_tag_var[0,i] = next_tag_var[0,i] + trans_val

                betas_t.append(log_sum_exp(next_tag_var))
            backward_var = torch.cat(betas_t).view(1, -1)
            beta.append(backward_var)

        log_beta = torch.cat(beta[::-1] , 0)
        return log_beta

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size+2).fill_(-10000.)
        init_vvars[0][ START_TAG ] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = [] # holds the backpointers for this step
            viterbivars_t = [] # holds the viterbi variables for this step

            for next_tag in xrange(self.tagset_size+2):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[ STOP_TAG ]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()

        best_path.reverse()
        return path_score, best_path

    def _marginal_decode(self, feats):
        # Use forward backward algorithm to calculate the marginal distribution
        # Decode according to the marginal distribution.
        _, log_alpha = self._forward_alg(feats)
        log_beta = self._backward_alg(feats)
        score = log_alpha+log_beta
        _, tags = torch.max(score, 1)
        tags = tags.view(-1).data.tolist()
        return score, tags

    def forward(self, feats):
    	score, _ = self._marginal_decode(feats)
    	return score

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable( torch.Tensor([0]) )

        tags = tags.data.numpy()
        tags = np.concatenate(([START_TAG], tags), axis=0)

        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[STOP_TAG, tags[-1]]
        return score

    def _get_neg_log_likilihood_loss(self, feats, tags):
        # nonegative log likelihood
        forward_score, _ = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score


    def _get_labelwise_loss(self, feats, tags):
    	'''
    	Training Conditional Random Fields for Maximum Labelwise Accuracy
    	'''
        # Get the marginal distribution
        score, _ = self._marginal_decode(feats)
        tags = tags.data.numpy()

        loss = autograd.Variable(torch.Tensor([0.]))
        Q = nn.Sigmoid()
        for tag, log_p in zip(tags, score):
            Pw = log_p[tag]
            if tag == 0:
                not_tag = log_p[1:]
            elif tag == len(log_p) - 1:
                not_tag = log_p[:tag]
            else:
                not_tag = torch.cat((log_p[:tag], log_p[tag+1:]))
            maxPw = torch.max(not_tag)
            loss = loss - Q(Pw - maxPw)
        return loss
