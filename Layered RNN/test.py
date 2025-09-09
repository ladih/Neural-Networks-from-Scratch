import rec_net
import matplotlib.pyplot as plt
import pickle

path = 'code.txt'
text = open(path, 'r').read()

def test1():
    # Test 1: mostly to see that the loss goes down in a reasonable way
    print("Test 1 started.")
    text1 = text[:15]
    vocab1 = rec_net.getvocab(text1)

    in_str = text1[:-1]
    tar_str = text1[1:]
    in_oh = rec_net.string_to_one_hot(in_str, vocab1)
    tar_oh = rec_net.string_to_one_hot(tar_str, vocab1)

    input_size, output_size = len(vocab1), len(vocab1)
    hidden_sizes = [32, 32]
    rnn_1 = rec_net.RNN(input_size, output_size, hidden_sizes)
    n_steps = 8000 # one step consists of a forward step and a backpropagation step
    learning_rate = 1e-2
    for i in range(n_steps):
        rnn_1.step(in_oh, tar_oh, learning_rate)
        if (i+1) % 500 == 0:
            loss = rnn_1.losscollector[-1]
            print("Processed", i+1, "/", n_steps, end=". ")
            print("Loss:", round(loss, 3))

    seed = text1[0]
    seed_oh = rec_net.string_to_one_hot(seed, vocab1)[0]
    gen_oh = rnn_1.generate(seed=seed_oh, n=40)
    gen_str = rec_net.one_hot_to_str(gen_oh, vocab1)
    print("\nOriginal string:")
    print(text1)
    print("\nGenerated string (40 characters):")
    print(seed + gen_str)

    plt.plot(rnn_1.losscollector)
    plt.title("Loss plot")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()

def test2():
    # Test 2: use the specialized train_text method and try to generate some text
    print("\nTest 2 started.")
    text2 = text[:200]
    vocab2 = rec_net.getvocab(text2)

    hidden_sizes = [32, 32, 32]
    input_size, output_size = len(vocab2), len(vocab2)
    rnn_2 = rec_net.RNN(input_size, output_size, hidden_sizes)

    learning_rate = 1e-2 # gets updated during training
    lossthr = 1e-3
    seqlength = 25 # number of characters read at a time
    n_fullreads = 10000
    rnn_2.train_text(text2, seqlength, vocab2, n_fullreads, learning_rate, lossthr)

    with open('saved_object.pkl', 'wb') as file: # saving object
        pickle.dump(rnn_2, file)


    with open('saved_object.pkl', 'rb') as file: # loading object
        rnn_2 = pickle.load(file)

    seed = text2[0]
    seed_onehot = rec_net.string_to_one_hot(seed, rnn_2.vocab)[0]

    n_chars = 1000 # number of generated characters
    gen_onehot = rnn_2.generate(seed_onehot, n_chars)
    gen_str = rec_net.one_hot_to_str(gen_onehot, rnn_2.vocab)

    print("\nOriginal string:\n")
    print(text2)

    print("\n\nSeed:", "'" + seed + "'\n")
    print("Generated text with", n_chars, "characters:\n")
    print(seed + gen_str)

test1()
test2()
