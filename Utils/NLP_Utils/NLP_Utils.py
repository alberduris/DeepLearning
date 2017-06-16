
# coding: utf-8

# # Some Utils for NLP
# 
# The document is structured:
# 
# [Brief explanation]<br>
# [Function]<br>
# [Example of use]<br>
# 
# All the functions comes with their correspondent docstring

# # Dependencies

# In[1]:

import tensorflow as tf
import codecs
import nltk
import re
import numpy as np
import collections
import string
import random
import warnings


# # Read Corpus Data
# 
# Just read your data :)

# In[2]:

def read_data(filename,encoding='utf-8'):
    '''
    Read the file named 'filename' and returns it as str
    args:
    filename: The file name
    encoding: default 'utf-8'
    return:
    A string with the contents of the file
    '''
    corpus_raw = u""
    with codecs.open(filename,'r','utf-8') as file:
        corpus_raw += file.read()
    return corpus_raw



# # Read several files and merge them into corpus

# In[4]:

def read_and_merge_files(filenames,encoding='utf-8'):
    '''
    Read and merge the files which names are contained in 'filenames' list and returns them as str
    args:
    filenames: A list of names in string
    encoding: default 'utf-8'
    return:
    A string with the contents of the files merged
    '''
    corpus_raw = u""
    for filename in filenames:
        print('Reading %s' % filename)
        with codecs.open(filename,'r','utf-8') as file:
            corpus_raw += file.read()
        print('Corpus in now %d characters long' % len(corpus_raw))
        print()
    return corpus_raw

#Clean corpus
def clean_corpus_az(corpus,caps='all'):
    '''
    Cleans corpus removing every character except [a-z,A-Z]
    args:
    corpus: A 'str' with the corpus to clean
    caps: Selects characters to retain by case
    'all' -> Lower and Upper case
    'lower' -> Just lower case
    'upper' -> Just upper case
    return:
    The cleaned corpus.
    '''
    assert type(corpus) is str, "corpus is not a string: %r" % corpus
    
    if(caps=='all'):
        clean = re.sub("[^a-zA-Z]"," ",corpus)
        clean = ' '.join(clean.split())
    elif(caps=='lower'):
        corpus = corpus.lower()
        clean = re.sub("[^a-z]"," ",corpus)
        clean = ' '.join(clean.split())
    elif(caps=='upper'):
        corpus = corpus.upper()
        clean = re.sub("[^A-Z]"," ",corpus)
        clean = ' '.join(clean.split())
    
    return clean

# # Extract sentences from corpus

# In[6]:

def extract_sentences(corpus):
    '''
    Extracts all the sentences present in a corpus. Uses NLTK tokenizer for english.
    args:
    corpus: A 'str' with the corpus to extract sentences from
    return:
    The corpus represented as a list of sentences.
    '''
    assert type(corpus) is str, "corpus is not a string: %r" % corpus
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(corpus)
    return raw_sentences


# # Extract word from corpus

# In[8]:

def extract_words(corpus):
    '''
    Extracts all the words present in a corpus. Removes everything except [a-z] & [A-Z]
    args:
    corpus: A 'str' with the corpus to extract words from
    return:
    The corpus represented as a list of words.
    '''
    assert type(corpus) is str, "corpus is not a string: %r" % corpus
    clean = re.sub("[^a-zA-z]"," ",corpus)
    words = clean.split()
    return words


# # Exract characters from corpus

# In[10]:

def extract_characters(corpus):
    '''
    Extracts all the characters in a corpus.
    args:
    corpus: A 'str' with the corpus to extract words from
    return:
    The corpus represented as a list of characters.
    '''
    chars = []
    for line in corpus:
        chars.extend(line)
    return chars



# # Extract unique word from corpus

# In[12]:

def extract_unique_words(corpus):
    '''
    Generates the words vocabulary of the corpus.
    Extracts every unique single words present in a corpus. Removes everything except [a-z] & [A-Z]
    args:
    corpus: A 'str' with the corpus to extract words from
    return:
    A list of str with all the unique present words in the corpus
    '''
    assert type(corpus) is str, "corpus is not a string: %r" % corpus
    clean = re.sub("[^a-zA-z]"," ",corpus)
    words = clean.split()
    return list(set(words))



# # Extract unique characters from corpus

# In[14]:

def extract_unique_characters(corpus):
    '''
    Generates the char vocabulary of the corpus.
    Extracts every unique single character present in a corpus. 
    args:
    corpus: A 'str' with the corpus to extract characters from
    return:
    A list of str with all the unique present characters in the corpus
    '''
    return list(set(corpus))



# # Vocab to ID & ID to Vocab DICTIONARIES

# In[16]:

def create_dictionaries(vocab):
    '''
    Creates Dictionary 'id_to_vocab' and Reverse Dictionary 'vocab_to_id' from a given vocab (chars or words)
    args:
    vocab: A list with every unique single word or character present in a corpus. Could be the output of 'extract_unique_words' or 'extract_unique_characters' functions.
    return:Dictionaries: [id_to_vocab,vocab_to_id]
    '''
    id_to_vocab = {i:ch for i,ch in enumerate(vocab)}
    vocab_to_id = {ch:i for i,ch in enumerate(vocab)}
    return id_to_vocab,vocab_to_id


# # Corpus words as integers

# In[19]:

def extract_corpus_as_integers(corpus,vocab_to_id,level='words'):
    '''
    Generate the integers representation of a corpus with a provided vocab_to_id dictictionary (that could be the output of 'create_dictionaries' function)
    args:
    corpus: The corpus as 'str' to generate the representation from
    vocab_to_id: A dictionary to map the representation {'vocab_element':'id'}
    level: Two possible values ['words','chars']. Representation of the corpus by words or by characters. Default 'words'
    return:
    The corpus as a list of int
    '''
    assert level == 'words' or level=='chars', 'level must be "words" or "chars"'
    corpus_as_int = []
    
    if(level == 'words'):
        clean = re.sub("[^a-zA-z]"," ",corpus)
        words = clean.split()

        for word in words:
            corpus_as_int.append(vocab_to_id[word])

    elif(level == 'chars'):
        chars = []
        for line in corpus:
            chars.extend(line)
        for char in chars:
            corpus_as_int.append(vocab_to_id[char])
            
    return corpus_as_int


# # Build dataset 
# 
# Google style

# In[22]:

#TODO
def build_dataset(words, vocabulary_size):
    '''
    Creates a dataset, given the corpus as a list of words and a vocabulary size representing the number of words to retain.
    Replaces unknown words with <unk> token.
    The dataset is the corpus represented by words as integers.
    args:
    words: The corpus as a list of words
    vocabulary_size: The number of words to retain
    returns:
    data: The corpus, as a list of words by its integer representation. '0' means 'unknown word'
    count: list that holds the counting with pairs like {'word':count}. Includes the <unk> count as first element.
    dictionary: A 'vocab_to_id' dictionary like {'word':id}. By construction, lower id means most common.
    reverse_dictionary An 'id_to_vocab' dictionary like {id,'word'}. By construction, lower id means most common.
    '''
    data = [] #Here we'll return the corpus as integers
    
    count = [['UNK', -1]] #Initialize the 'unknown' counter with -1
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1)) #Store in count pairs like {'word':appearances}
    
    #Creates a dictionary with the unique words like {'word':id}
    #By construction, the most common words have the lowest id
    dictionary = dict()
    #Construction: Iterates the count from most common to uncommon. The dictionary length increases at each step.
    for word, _ in count:
        dictionary[word] = len(dictionary) 
  
    #Iterates over the words
    unk_count = 0
    for word in words:
        if word in dictionary: #If word is in dictionary, then is a 'vocabulary_size' common word
            index = dictionary[word] #The index is given by the dictionary
        else:
            index = 0  #Add to the first entry of the list, where we store unknown words => dictionary['UNK'] 
            unk_count += 1 #Update the unknown word count
        data.append(index) #Append to data
        
    #Now data is a representation of the corpus, by words, as integers. Note that the presence of '0' means 'unknown word'
    
    
    count[0][1] = unk_count #Update the unknown count in 'count'
    
    #Creates dictionary (vocab_to_id) and reverse dictionary
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    
    return data, count, dictionary, reverse_dictionary
    
# # Generate Word2Vec Batches

# In[31]:

def w2v_generate_batch(data,batch_size=16,num_skips=1,skip_window=1):
    """Return batch and label from data
    args:
    data: A list of 'int'
     
    """ 
    global data_index
    data_index = 0
    assert batch_size % num_skips == 0 
    assert num_skips < 2*skip_window 
    
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    
    span = 2 * skip_window + 1 # Total length, target + context => [skip_window target skip_window]
    buffer = collections.deque(maxlen=span) #A list-like sequence with maxlen, if more appended, the first dissappears
    
    #Add the the words to buffer [skip_window target skip_window]
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    #A más skips menos vueltas
    for i in range(batch_size // num_skips): 
        target = skip_window #Target at the center
        targets_to_avoid = [skip_window]
        
        for j in range(num_skips):#For each skip
            while target in targets_to_avoid:
                target = np.random.randint(0,span-1)
            targets_to_avoid.append(target)
            
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j,0] = buffer[target]
        
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
        
    #Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch,labels


def rnn_minibatch_sequencer(raw_data, batch_size, sequence_size, nb_epochs,num_features,num_classes):
    """
    Divides the data into batches of sequences so that all the sequences in one batch
    continue in the next batch. This is a generator that will keep returning batches
    until the input data has been seen nb_epochs times. Sequences are continued even
    between epochs, apart from one, the one corresponding to the end of raw_data.
    The remainder at the end of raw_data that does not fit in an full batch is ignored.
    :param raw_data: the training text
    :param batch_size: the size of a training minibatch
    :param sequence_size: the unroll size of the RNN
    :param nb_epochs: number of epochs to train on
    :return:
        x: one batch of training sequences
        y: on batch of target sequences, i.e. training sequences shifted by 1
        epoch: the current epoch number (starting at 0)
    """
    data = np.array(raw_data)
    data_len = data.shape[0]
    
    #Number of batches per epoch
    # using (data_len-1) because we must provide for the sequence shifted by 1 too <- Not want X without Y
    nb_batches = (data_len - 1) // (batch_size * sequence_size) 

    assert nb_batches > 0, "Not enough data, even for a single batch. Try using a smaller batch_size."
    
    #How many train data will be used - Divisible by (batch_size*sequence_size)
    rounded_data_len = nb_batches * batch_size * sequence_size
    
    #Divide data into batch size sequences of data
    xdata = np.reshape(a=data[0:rounded_data_len], newshape=[batch_size, nb_batches * sequence_size, num_features])
    #Y data is just +1
    ydata = np.reshape(a=data[1:rounded_data_len + 1], newshape=[batch_size, nb_batches * sequence_size, num_classes])
    
    for epoch in range(nb_epochs): #For each epoch
        for batch in range(nb_batches): #For each batch inside epoch
            
            
            #Fragment the sequence into [sequence_size] slices
            x = xdata[:, batch * sequence_size:(batch + 1) * sequence_size]
            y = ydata[:, batch * sequence_size:(batch + 1) * sequence_size]
            
            #To not lose the sequence between epochs
            x = np.roll(x, -epoch, axis=0)  # to continue the text from epoch to epoch (do not reset rnn state!)
            y = np.roll(y, -epoch, axis=0)
            
            yield x, y, epoch
  


#################################################BATCH MANAGER#################################################

#@TODO: num_epochs
#@TODO: generate_batch
#@TODO: one_hot

class BatchManager(object):

    def __init__(self):
        self._filename = ""

        self._text = u""
        self._text_size = -1
        
        self._vocab = list()
        self._vocab_size = -1
        self._vocab_strategy = 'all'

        self._mode = 'dense'
        
        self._batch_size = 16
        self._seq_length = 50
        self._overlap = 25

        self._num_batches = -1

    def set_params(self,filename,batch_size,seq_length,overlap,vocab_strategy,mode):

        if(filename is not None):
            self._filename = filename
            #Read corpus
            corpus_raw = u""
            with codecs.open(filename,'r','utf-8') as file:
                corpus_raw += file.read()
                self._text = corpus_raw
                self._text_size = len(corpus_raw)

        if(batch_size is not None):
            self._batch_size = batch_size
        if(seq_length is not None):
            self._seq_length = seq_length
        if(overlap is not None):
            self._overlap = overlap
        if(vocab_strategy is not None):
            self._vocab_strategy = vocab_strategy
            self.create_vocab(self,strategy=self._vocab_strategy)

        if(mode is not None):
            self._mode = mode

        if(self._batch_size is not None and self._seq_length is not None and self._overlap is not None):
            self.get_num_batches()


    def get_params(self):

        print('@params:')
        print('[filename : "%s"]' % self._filename)
        print('[text : "%s"]' % self._text[0:10])
        print('[text_size : %s]' % self._text_size)
        print('[vocab : %s]' % self._vocab[0:5])
        print('[vocab_size : %s]' % self._vocab_size)
        print('[batch_size : %s]' % self._batch_size)
        print('[seq_length : %s]' % self._seq_length)
        print('[overlap : %s]' % self._overlap)
        print('[num_batches : %s]' % self._num_batches)



    def create_vocab(self,filename, strategy = 'all'):

        if(self._text_size == 0):
            #Read corpus
            corpus_raw = u""
            with codecs.open(filename,'r','utf-8') as file:
                corpus_raw += file.read()
                self._text = corpus_raw
                self._text_size = len(corpus_raw)

        if(strategy == 'all'):
            vocab = list(set(self._text))
        elif(strategy == 'az'):
            corpus_raw = self._text.lower()
            clean = re.sub("[^a-z]"," ",corpus_raw)
        elif(strategy == 'az_num'):
            corpus_raw = self._text.lower()
            clean = re.sub("[^a-z0-9]"," ",corpus_raw)


        self._vocab = vocab
        self._vocab_size = len(vocab)
        return vocab


    def read_data(self):

        if(self._text_size == 0):
            warnings.warn("Text not defined - Halting. Define a file before proceed. Use BatchManager.set_params()", UserWarning)
            sys.exit(0)
        if(len(self._vocab) == 0):
            warnings.warn("Vocabulary not defined. Defining default vocabulary. You can define vocabulary with BatchManager.create_vocab()", UserWarning)
            self.create_vocab(filename)

        
        text = self._text
        text = self.vocab_encode(text)

        for start in range(0, self._text_size - self._seq_length, self._overlap):
            xData = text[start: start + self._seq_length]
            xData += [0] * (self._seq_length - len(xData))

            yData = text[start+1: start + self._seq_length+1]
            yData += [0] * (self._seq_length - len(yData))
            yield xData,yData

    
    #Generate batches
    def read_batch(self,stream):
        batchX = []
        batchY = []
        for elemX,elemY in stream:
            batchX.append(elemX)
            batchY.append(elemY)
            if len(batchX) == self._batch_size:
                yield batchX,batchY
                batchX = []
                batchY = []
        #yield batchX,batchY

    def generate_batches(self,num_epochs=1):
        for epoch in range(num_epochs):
            for batchX,batchY in self.read_batch(self.read_data()):
                yield batchX,batchY,epoch


    def vocab_encode(self,text):
        return [self._vocab.index(x) + 1 for x in text if x in self._vocab]


    def vocab_decode(self,list_ids):
        return ''.join([self._vocab[x - 1] for x in list_ids])


    def stats(self):



        print('The corpus has %d characters' %self._text_size)
        print('Configuration:\n[batch_size : %d]\n[seq_length : %d]\n[overlap : %d]\n' % (self._batch_size, self._seq_length, self._overlap))

        print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past and overlapping %d steps' 
            %(self.get_num_batches(),self._batch_size,self._seq_length, (self._seq_length-self._overlap)))


    def get_num_batches(self):
        self._num_batches = (self._text_size-1) // (self._batch_size * (self._seq_length - (self._seq_length-self._overlap)))
        return self._num_batches

#################################################END-STANDFORD#################################################

            
#################################################ONE-HOT-SPARSE#################################################



vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
#Unicode code point for a one-character string
first_letter = ord(string.ascii_lowercase[0])

'''
@post: Dado un char devuelve su identificar basado en Unicode code point
'''
def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
    return 0

'''
@post: Dado un id propio, devuelve el char asociado
'''
def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

batch_size=1
num_unrollings=3

###ONE HOT
class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        
        #Params
        self._text = text #The raw text as str - Output of NLP_Utils.read_data()
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings

        #The size of each fragment (batches) the data can be divided in
        #Ie: A text w/ 100 characters and batch size 2 gives 2 fragments of size 50. Segment is 50.
        segment = self._text_size // batch_size

        #The starting position for each fragment
        self._cursor = [ offset * segment for offset in range(batch_size)]
        
        #First characters for each fragment or batch instance
        self._last_batch = self._next_batch()


    def _next_batch(self):
        """Extracts one character for each batch instance"""
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generates the subsequent (seq_length) characters for each fragment or batch instance"""
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
          batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

    def characters(self,probabilities):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (mostl likely) character representation."""
        return [id2char(c) for c in np.argmax(probabilities, 1)]

    def batches2string(self,batches):
        """Convert a sequence of batches back into their (most likely) string
        representation."""
        s = [''] * batches[0].shape[0]
        for b in batches:
            s = [''.join(x) for x in zip(s, BatchGenerator.characters(self,b))]
        return s


    '@post: Log-probability of the true labels in a predicted batch'
    def logprob(self,predictions, labels):
        
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

    '@post: Sample one element from a distribution assumed to be an array of normalized probabilities.'
    def sample_distribution(self,distribution):
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(distribution)):
            s += distribution[i]
            if s >= r:
                return i
        return len(distribution) - 1

    '@post: Turn a (column) prediction into 1-hot encoded samples.'
    def sample(self,prediction):
        p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
        p[0, BatchGenerator.sample_distribution(self,prediction[0])] = 1.0
        return p

    '@post: Generate a random column of probabilities.'
    def random_distribution(self):
        b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
        return b/np.sum(b, 1)[:,None]



########SPARSE

class BatchGeneratorSparse(object):
    def __init__(self, text, batch_size, num_unrollings):
        
        #Params
        self._text = text #The raw text as str - Output of NLP_Utils.read_data()
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings

        #The size of each fragment (batches) the data can be divided in
        #Ie: A text w/ 100 characters and batch size 2 gives 2 fragments of size 50. Segment is 50.
        segment = self._text_size // batch_size

        #The starting position for each fragment
        self._cursor = [ offset * segment for offset in range(batch_size)]
        
        #First characters for each fragment or batch instance
        self._last_batch = self._next_batch_sparse()




    def _next_batch_sparse(self):
        """Extracts one character for each batch instance"""
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, 1), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, 0] = char2id(self._text[self._cursor[b]])
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next_sparse(self):
        """Generates the subsequent (seq_length) characters for each fragment or batch instance"""
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
          batches.append(self._next_batch_sparse())
        self._last_batch = batches[-1]
        return batches

    
    def characters_sparse(self,char_ids):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (mostl likely) character representation."""
        return [id2char(c) for c in char_ids]

    def batches2string_sparse(self,batches):
        """Convert a sequence of batches back into their (most likely) string
        representation."""
        s = [''] * batches[0].shape[0]
        for b in batches:
            s = [''.join(x) for x in zip(s, BatchGeneratorSparse.characters_sparse(self,b))]

        return s


    '@post: Log-probability of the true labels in a predicted batch'
    def logprob(self,predictions, labels):
        
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

    '@post: Sample one element from a distribution assumed to be an array of normalized probabilities.'
    def sample_distribution(self,distribution):
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(distribution)):
            s += distribution[i]
            if s >= r:
                return i
        return len(distribution) - 1

    '@post: Turn a (column) prediction into 1-hot encoded samples.'
    def sample(self,prediction):
        p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
        p[0, BatchGenerator.sample_distribution(self,prediction[0])] = 1.0
        return p

    '@post: Generate a random column of probabilities.'
    def random_distribution(self):
        b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
        return b/np.sum(b, 1)[:,None]


#################################################ONE-HOT-SPARSE#################################################