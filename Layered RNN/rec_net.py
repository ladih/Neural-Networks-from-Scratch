import numpy as np

class RNN():
    def __init__(self, input_size, output_size, hidden_sizes):
        self.input_size = input_size # number of classes, i.e. len of the one-hot encoded input label
        self.output_size = input_size # same but for output (oftne same as input_size)
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        self.weightmatrices = []
        low, high = -0.4, 0.4 # range of the uniform distribution that is used to initialize weight matrices
        # initialize weightmatrices
        self.weightmatrices.append(np.random.uniform(low, high, (input_size, hidden_sizes[0]))) # weights from input to first hidden
        for i in range(len(hidden_sizes[:-1])):
            self.weightmatrices.append(np.random.uniform(low, high, (hidden_sizes[i], hidden_sizes[i]))) # weights from hidden layer i to itself
            self.weightmatrices.append(np.random.uniform(low, high, (hidden_sizes[i], hidden_sizes[i+1]))) # weights from hidden layer i to hidden layer i+1
        self.weightmatrices.append(np.random.uniform(low, high, (hidden_sizes[-1], hidden_sizes[-1]))) # weights from last hidden to itself
        self.weightmatrices.append(np.random.uniform(low, high, (hidden_sizes[-1], output_size))) # weights from last hidden to output

        self.biases = []
        # initialize biases
        for size in hidden_sizes:
            self.biases.append(np.zeros((1, size))) # biases for hidden layers
        self.biases.append(np.zeros((1, output_size))) # bias for output

        self.hidden_states = []
        # initialize hidden states
        for size in hidden_sizes:
            self.hidden_states.append(np.zeros((1, size))) # initialize hidden states

        self.losscollector = []

    def step(self, inputs, targets, learning_rate, hprevs=0):
        """Forward and backpropagation step. Updates weights.
           inputs and targets should be a lists of one-hot encoded labels."""
        inputs = [np.array(sublist).reshape(1, -1) for sublist in inputs] # reshape input
        targets = [np.array(sublist).reshape(1, -1) for sublist in targets]   

        if hprevs == 0: # if no hidden states at t=-1 are given
            for i, s in enumerate(self.hidden_sizes):
                self.hidden_states[i] = np.zeros((1, s))
        hprevs = self.hidden_states

        # forward step
        loss = 0
        raws_t = []
        ps_t = []
        for t in range(len(inputs)):
            yraw = [inputs[t]] # raw outputs from layers (before weighting them)
            for i in range(self.num_layers):
                hidden = np.tanh(yraw[-1] @ self.weightmatrices[i*2]
                                    + self.hidden_states[i] @ self.weightmatrices[i*2+1]
                                    + self.biases[i]) # hidden state
                self.hidden_states[i] = hidden
                yraw.append(hidden) # collect hidden states, i.e. raw outputs from the hidden layers
            y = yraw[-1] @ self.weightmatrices[-1] + self.biases[-1] # output from last layer
            ps = np.exp(y) / np.sum(np.exp(y)) # softmax activation
            ps_t.append(ps) # collect softmax probabilities
            raws_t.append(yraw) # collect the raw outputs for the current time step
            loss += -np.log(ps[0][np.argmax(targets[t])])
        self.losscollector.append(loss)

        # backpropagation step
        next_dhraws = [] # derivative at next hidden layer (which doesn't exist at the start, so initialize to zero)
        for size in self.hidden_sizes:
            next_dhraws.append(np.zeros((1, size)))
        dW_mats = [] # weight update matrices
        for mat in self.weightmatrices:
            dW_mats.append(np.zeros_like(mat))
        dbiases = [] # bias updates
        for b in self.biases:
            dbiases.append(np.zeros_like(b))
        for t in reversed(range(len(inputs))):
            bp_y = np.copy(ps_t[t])
            bp_y[0][np.argmax(targets[t])] -= 1 # backpropagation at output layer
            dWlast = raws_t[t][-1].T @ bp_y
            dW_mats[-1] += dWlast  # accumulating weight updates for weights from last hidden to output layer
            dbiases[-1] += bp_y
            for l in range(self.num_layers):
                bp = bp_y @ self.weightmatrices[-2*l-1].T + next_dhraws[-l-1] @ self.weightmatrices[-2*l-2].T # backprop through the hidden layers
                bp_tanh = (1 - raws_t[t][-l-1] * raws_t[t][-l-1]) * bp # backprop through tanh
                if t == 0:
                    if hprevs == 0: # if no hidden state before t=0 is given
                        dWh = np.zeros_like(self.weightmatrices[-2*l-2])
                    else:
                        dWh = hprevs[-l-1].T @ bp_tanh
                else:
                    dWh = raws_t[t-1][-l-1].T @ bp_tanh # weight update for hidden weights (uses raw hidden output from previous time step)
                dWx = raws_t[t][-l-2].T @ bp_tanh # weight update for weights from imput to first hidden
                dW_mats[-2*l-2] += dWh # accumulate weight updates
                dW_mats[-2*l-3] += dWx
                dbiases[-l-2] += bp_tanh
                next_dhraws[-l-1] = bp_tanh
                bp_y = bp_tanh

        clipthr = 2
        # clip weight updates (to prevent exploding gradients)
        for i in range(len(dW_mats)):
            dW_mats[i] = np.clip(dW_mats[i], -clipthr, clipthr, out=dW_mats[i])
        for i in range(len(dbiases)):
            dbiases[i] = np.clip(dbiases[i], -clipthr, clipthr, out=dbiases[i])

        # update weights
        for i in range(len(self.weightmatrices)):
            self.weightmatrices[i] += -learning_rate * dW_mats[i]
        for i in range(len(self.biases)):
            self.biases[i] += -learning_rate * dbiases[i]

        return [raws_t[-1][-i-1] for i in range(self.num_layers)] # returns last hidden states, in case these should be used for the next call to step

    def train_text(self, text, seqlength, vocab, n_fullreads, learning_rate, lossthr):
        """Training method specialized for text. text should be a string.
            seqlength is how many characters of the text that are passed
            to the step function each time."""
        self.vocab = vocab # need this attribute so that we can generate text later from a saved rnn object
        tlen = len(text)
        sequences_pertext = (tlen-1) // seqlength
        reads = 0 # number of reads through the full text
        stoptraining = 0
        p, n = 0, 0 # pointer starts at beginning of the text
        print("text has", len(text), "characters with a total of", len(vocab), "unique characters.")
        print("Training started. Reading", seqlength, "characters at a time, which gives ", end="")
        print(sequences_pertext, "sequences per text.")
        while True:

            if p+1+seqlength >= len(text): # if end of target string is past len(text)
                for i, size in enumerate(self.hidden_sizes):
                    self.hidden_states[i] = np.zeros((1, size)) # reset hidden states
                p = 0 # reset pointer
                reads += 1


                mloss = np.sum(self.losscollector[-sequences_pertext:]) / sequences_pertext

                # try to catch the model at low loss
                if mloss < 0.1:
                    learning_rate = 1e-3
                elif mloss < 1:
                    learning_rate = 5e-3

                if reads % 1000 == 0:
                    print("Number of reads:", reads, "/", n_fullreads, end=", ")
                    print("meanloss:", round(mloss, 3))
                    if mloss < lossthr:
                        print("meanloss is <", lossthr, end=", training stops.")
                        stoptraining = 1
                    if reads == n_fullreads:
                        print("Have read", reads, "times, training stops.")
                        stoptraining = 1
                self.losscollector = [] # reset loss memory

            if stoptraining == 1:
                break

            inpstr = text[p:p+seqlength]
            targetstr = text[p+1:p+1+seqlength]
            inp_oh = string_to_one_hot(inpstr, vocab)
            out_oh = string_to_one_hot(targetstr, vocab)
            self.step(inp_oh, out_oh, learning_rate, hprevs=1)

            p += seqlength
            n += 1 # iteration counter... isn't used for anything

    def generate(self, seed, n, init_to_zero=1):
        """Generates n one-hot encoded labels.
           seed should be a one-hot encoded label"""
        hprevs = self.hidden_states
        if init_to_zero == 1:
            for i, s in enumerate(self.hidden_sizes):
                hprevs[i] = np.zeros((1, s))
        cur = np.array(seed).reshape(1, -1)
        ws = self.weightmatrices
        bs = self.biases
        gens = []
        for t in range(n):
            for i in range(len(ws)//2):
                cur = np.tanh(cur @ ws[2*i] + hprevs[i] @ ws[2*i+1] + bs[i])
                hprevs[i] = cur
            cur = cur @ self.weightmatrices[-1] + bs[-1]
            cur = np.exp(cur) / np.sum(np.exp(cur)) # softmax activation
            ind = np.random.choice(range(len(cur[0])), p=cur.ravel())
            for j in range(cur.shape[1]):
                if j == ind:
                    cur[0][j] = 1
                else:
                    cur[0][j] = 0
            gens.append(cur)
        return gens

# functions

def string_to_one_hot(s, vocab):
    """input: a string s and a vocab (list of characters)
       outut: list of one-hot encoded characters corresponding to s and vocab"""
    char_to_id = {c:i for i, c in enumerate(vocab)}
    n = len(vocab)
    ohs = [[0 for _ in range(n)] for _ in range(len(s))]
    for i, c in enumerate(s):
        ind = char_to_id[c]
        ohs[i][ind] = 1
    return ohs

def one_hot_to_str(ohs, vocab):
    """input: ohs should be a list of one-hot encoded characters corresponding to vocab,
              vocab should be a list of characters
       output: the string corresponding to ohs and vocab"""
    id_to_char = {i:l for i, l in enumerate(vocab)}
    ind_vec = [np.argmax(o) for o in ohs]
    str_vec = [id_to_char[i] for i in ind_vec]
    return ''.join(str_vec)

def ps_to_labels(ps):
    """input: list of softmax predictions
       output: the corresponding one-hot encoded labels"""
    labels = []
    for p in ps:
        n_classes = p.shape[1]
        one_hot = np.zeros((1, n_classes))
        i = np.random.choice(range(len(n_classes)), p=cur.ravel()) # index based on the probabilities of all classes (to add some randomness)
        one_hot[0][i] = 1
        labels.append(one_hot)
    return labels

def getvocab(text):
    vocab = list(set(text))
    vocab.sort()
    return vocab
