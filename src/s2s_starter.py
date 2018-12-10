import torch
import numpy as np
from torch import nn


# class for our vanilla seq2seq
class S2S(nn.Module):
    def __init__(self, d_char, d_hid, len_voc):

        super(S2S, self).__init__()
        self.d_char = d_char
        self.d_hid = d_hid
        self.len_voc = len_voc

        # embeddings
        self.char_embs = nn.Embedding(len_voc, d_char)

        # encoder and decoder RNNs
        self.encoder = nn.RNN(d_char, d_hid, num_layers=1, batch_first=True)
        self.decoder = nn.RNN(d_char, d_hid, num_layers=1, batch_first=True)
        
        # output layer (softmax will be applied after this)
        self.out = nn.Linear(d_hid, len_voc)

    
    # perform forward propagation of S2S model
    def forward(self, inputs, outputs):
        
        bsz, max_len = inputs.size()

        # get embeddings of inputs
        embs = self.char_embs(inputs)

        # encode input sentence and extract final hidden state of encoder RNN
        _, final_enc_hiddens = self.encoder(embs)
                
        # initialize decoder hidden to final encoder hiddens
        hn = final_enc_hiddens
        
        # store all decoder states in here
        decoder_states = torch.zeros(max_len, bsz, self.d_hid)

        # now decode one character at a time
        for idx in range(max_len):

            # store the previous character if there was one
            prev_chars = None 
            if idx > 0:
                prev_chars = outputs[:, idx - 1] # during training, we use the ground-truth previous char

            # feed previous hidden state and previous char to decoder to get current hidden state
            if idx == 0:
                decoder_input = torch.zeros(bsz, 1, self.d_char)

            # get previous ground truth char embs
            else:
                decoder_input = self.char_embs(prev_chars)
                decoder_input = decoder_input.view(bsz, 1, self.d_char)

            # feed to decoder rnn and store hidden state in decoder states
            _, hn = self.decoder(decoder_input, hn)
            decoder_states[idx] = hn 

        # now do prediction over decoder states (reshape to 2d first)
        decoder_states = decoder_states.transpose(0, 1).contiguous().view(-1, self.d_hid)
        decoder_preds = self.out(decoder_states)
        decoder_preds = torch.nn.functional.log_softmax(decoder_preds, dim=1)

        return decoder_preds


    # given a previous character and a previous hidden state
    # produce a probability distribution over the entire vocabulary
    def single_decoder_step(self, prev_char, prev_hid):
        if prev_char is not None:
            decoder_input = self.char_embs(prev_char).expand(1, 1, self.d_char)
        else:
            decoder_input = torch.zeros(1, 1, self.d_char)

        # feed prev hidden state and prev char to decoder to get current hidden state
        _, hn = self.decoder(decoder_input, prev_hid)

        # feed into output layer and apply softmax to get probability distribution over vocab
        pred_dist = self.out(hn.transpose(0, 1).contiguous().view(-1, self.d_hid))
        pred_dist = torch.nn.functional.log_softmax(pred_dist, dim=1)
        return pred_dist.view(-1), hn


    # greedy search for one input sequence (bsz = 1)
    def greedy_search(self, seq):

        bsz, max_len = seq.size()

        output_seq = [] # this will contain our output sequence
        output_prob = 0. # and this will be the probability of that sequence

        # get embeddings of inputs
        embs = self.char_embs(seq)

        # encode input sentence and extract final hidden state of encoder RNN
        _, final_enc_hidden = self.encoder(embs)

        # initialize decoder hidden to final encoder hidden
        hn = final_enc_hidden
        prev_char = None
        
        # now decode one character at a time
        for idx in range(max_len):

            pred_dist, hn = self.single_decoder_step(prev_char, hn)
            _, top_indices = torch.sort(-pred_dist) # sort in descending order (log domain)

            # in greedy search, we will just use the argmax prediction at each time step
            argmax_pred = top_indices[0]
            argmax_prob = pred_dist[argmax_pred]
            output_seq.append(argmax_pred.numpy())
            output_prob += argmax_prob

            # now use the argmax prediction as the input to the next time step
            prev_char = argmax_pred

        return output_prob, output_seq


    # beam search for one input sequence (bsz = 1)
    # YOUR JOB: FILL THIS OUT!!! SOME SPECIFICATION:
    # input: seq (a single input sequence)
    # output: beams, a list of length beam_size containing the most probable beams
    #         each element of beams is a tuple whose first two elements are
    #         the probability of the sequence and the sequence itself, respectively.
    def beam_search(self, seq, beam_size=5):

        bsz, max_len = seq.size()

        # get embeddings of inputs
        embs = self.char_embs(seq)

        # encode input sentence and extract final hidden state of encoder RNN
        _, final_enc_hidden = self.encoder(embs)
        
        # this list will contain our k beams
        # you might find it helpful to store three things:
        #       (prob of beam, all chars in beam so far, prev hidden state)
        beams = [(0.0, [], final_enc_hidden)]

        # decode one character at a time
        for idx in range(max_len):

            # add all candidate beams to the below list
            # later you will take the k most probable ones where k = beam_size
            beam_candidates = []

            for b in beams:
                curr_prob, seq, prev_h = b

                if len(seq) == 0:
                    prev_char = None
                else:
                    prev_char = seq[-1]

        # fill out the rest of the beam search!
        # the greedy_search code might be helpful to look at and understand!

                pred_dist, prev_h = self.single_decoder_step(prev_char, prev_h)
                _, top_indices = torch.sort(-pred_dist)  # sort in descending order (log domain)

                # in beam search, we will get all candidates
                argmax_preds = top_indices

                # expand each candidate and add to list of candidates
                for argmax_pred in argmax_preds:
                    output_prob = curr_prob
                    output_prob += pred_dist[argmax_pred]

                    output_seq = list(seq)
                    output_seq.append(argmax_pred)

                    beam_candidates.append((output_prob, output_seq, prev_h))

            # sort and add top candidate to beam
            sorted_beams = sorted(beam_candidates, key=lambda tup: tup[0])
            beams.append(sorted_beams[0])

            beam_candidates.clear()

        return beams[1:]

    def beam_check(self, seq):
        beams = self.beam_search(seq.expand(1, seq.size()[1]), beam_size=1)
        greedy_prob, greedy_out = self.greedy_search(seq.expand(1, seq.size()[1]))
        beam_prob = beams[0][0]
        beam_out = [np.array(c) for c in beams[0][1]]

        #beams[0][0] == greedy_prob
        #torch.allclose(beams[0][0], greedy_prob
        if beams[0][0] == greedy_prob and greedy_out == beam_out:
            return True
        else: 
            return False
