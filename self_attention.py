import numpy as np
import torch
import tensorflow as tf
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

def self_attention(sentence):
  """
  Performs self-attention on the given queries, keys, and values.

  Args:
    queries: A tensor of shape (batch_size, seq_len, hidden_dim).
    keys: A tensor of shape (batch_size, seq_len, hidden_dim).
    values: A tensor of shape (batch_size, seq_len, hidden_dim).

  Returns:
    A tensor of shape (batch_size, seq_len, hidden_dim).
  """

  tokenized_words = word_tokenize(sentence)

  # Create a vocabulary and map tokens to numerical IDs
  vocab = {word: idx for idx, word in enumerate(set(tokenized_words))}

  # Convert tokens to numerical IDs
  numerical_tokens = [vocab[word] for word in tokenized_words]

  # Convert to a PyTorch tensor
  tensor_sentence = torch.tensor(numerical_tokens, dtype=torch.int64)

  sentence_length = len(sentence.split())

  print("Sentence length", sentence_length)


  print(tensor_sentence)
  torch.manual_seed(123)
  embed = torch.nn.Embedding(sentence_length, 24)
  embedded_sentence = embed(tensor_sentence)

  torch.manual_seed(123)

  d = embedded_sentence.shape[1]

  d_q, d_k, d_v = 24, 24, 28

  W_query = torch.nn.Parameter(torch.rand(d_q, d))
  W_key = torch.nn.Parameter(torch.rand(d_k, d))
  W_value = torch.nn.Parameter(torch.rand(d_v, d))

  keys = W_key.matmul(embedded_sentence.T).T
  values = W_value.matmul(embedded_sentence.T).T

  print("keys.shape:", keys.shape)
  print("values.shape:", values.shape)

  query = W_query.matmul(embedded_sentence.T).T
  omega = query.matmul(keys.T)

  scores = query.matmul(keys.T) / np.sqrt(d_k)
  # Apply the softmax function to the attention scores.
  attention = F.softmax(scores,dim=0)

  # Multiply the attention weights with the values.
  output = attention.matmul(values)

  return output

def main():
  # Create the queries, keys, and values.
  queries = np.random.randn(3, 5, 10)
  keys = np.random.randn(3, 5, 10)
  values = np.random.randn(3, 5, 10)
  sentence = "when I lived in France"
  # Perform self-attention.
  attention_values = self_attention(sentence)

  # Print the attention values.
  print(attention_values)

if __name__ == "__main__":
  main()
