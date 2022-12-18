import string
import numpy as np
import torch


def word_to_one_hot_vector(word_string, alphabet=string.ascii_lowercase + string.digits + ".,?!-\'_"):
    if len(word_string) >= 15:
        fixed_length_word_string = word_string[:15]
    else:
        fixed_length_word_string = word_string
        for i in range(len(word_string), 15):
            fixed_length_word_string += '_'
    vector = [[0 if char != letter else 1 for char in alphabet]
              for letter in fixed_length_word_string]
    print(len(alphabet))
    return torch.flatten(torch.tensor(vector, dtype=torch.float))


def get_one_hot_vectors(x):
    one_hot_vectors_of_all_words = []
    for batch_unit in x:
        one_hot_vectors_for_a_word = []
        for temp_word in batch_unit:
            one_hot_vectors_for_a_word.append(torch.unsqueeze(word_to_one_hot_vector(temp_word), dim=0))
        print(one_hot_vectors_for_a_word[0].shape)
        temp_one_hot = torch.cat(one_hot_vectors_for_a_word, dim=0)
        one_hot_vectors_of_all_words.append(torch.unsqueeze(temp_one_hot, dim=0))
    return torch.cat(one_hot_vectors_of_all_words, dim=0)


def get_char_embedding(x):
    one_hot_vectors_of_all_words = get_one_hot_vectors(x)
    print(one_hot_vectors_of_all_words.shape)

    conv_layer = torch.nn.Conv1d(in_channels=one_hot_vectors_of_all_words.shape[1],
                                 out_channels=one_hot_vectors_of_all_words.shape[1], kernel_size=5)
    temp_vector = one_hot_vectors_of_all_words  # torch.unsqueeze(one_hot_vectors_of_all_words, dim=1)
    print("-> ", temp_vector.shape)
    temp_vector = conv_layer(temp_vector)
    print("---> ", temp_vector.shape)
    pool_layer = torch.nn.MaxPool1d(kernel_size=10)
    temp_vector = pool_layer(temp_vector)
    print("------> ", temp_vector.shape)

    """for v in one_hot_vectors_of_all_words:
        conv_layer = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(3, 3))
        temp_vector = torch.unsqueeze(v, dim=0)
        temp_vector = conv_layer(temp_vector)
        print(temp_vector.shape)
        # pool_layer = torch.nn.MaxPool1d(kernel_size=len())"""


X = [["abcd", "qefq", "rggrqab"], ["gwwf", "btbwbb", "tntbwb"]]  # batch_size = 3
get_char_embedding(X)

