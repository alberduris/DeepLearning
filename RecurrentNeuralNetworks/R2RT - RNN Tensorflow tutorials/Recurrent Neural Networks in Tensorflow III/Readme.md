# Recurrent Neural Networks in Tensorflow III

Several RNN models suited for Variable Sequence Length data are implemented in Tensorflow, including a simple Seq2Seq model.

## RNN_VariableLength_Text_Classifier.ipynb

In this Notebook, we’ll use Tensorflow to construct an RNN that operates on input sequences of variable lengths. We’ll use this RNN to classify bloggers by age bracket and gender using sentence-long writing samples. One time step will represent a single word, with the complete input sequence representing a single sentence. The challenge is to build a model that can classify multiple sentences of different lengths at the same time. <a href="https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html">Source</a>

## RNN_VariableLength_Text_Classifier_Bucketing.ipynb

In this Notebook, we'll implement Bucketing technique. <a href="https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html#improving-training-speed-using-bucketing">Source</a>

## Seq2Seq_RNN_VariableLength_Text_Classifier_Bucketing.ipynb

In this Notebook, we'll extend our sequence classification model to do sequence-to-sequence learning. We’ll use the same dataset, but instead of having our model guess the author’s age bracket and gender at the end of the sequence (i.e., only once), we’ll have it guess at every timestep. <a href="https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html#a-basic-model-for-sequence-to-sequence-learning">Source</a>