
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
import sys


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


class BatchManager(object):
    '''
    BatchManager is a wrapper for functions which purpose is to ease the process of feeding data to Neural Networks. Specifically RNNs and for TensorFlow.

    Available functions:


    '''

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

        self._configured = False

    def set_params(self,filename,batch_size,seq_length,overlap,vocab_strategy,mode):
        '''
        Mandatory function that should be called first of all and configures the BatchManager behaviour according to the params.
        args:
        filename: The name/path of the file to extract the corpus from as 'str'
        batch_size: The size of each batch. 'int'
        seq_length: Also known as num_unrollings or time_steps. For RNNs, the length of the sequence. 'int'
        overlap: Defines the overlapping between successive sequences. If overlaps equals seq_length means no overlapping. 'int'
        vocab_strategy: Defines how the vocabulary will be extracted. 
            -'all' : No cleaning is done. All characters and strange characters are hold. 
            -'az' : All characters are lower cased and only [a-z] characters are hold.
            -'az_num' : All characters are lower cased and only [a-z] & [0-9] characters are hold.
        mode: Defines the representation of the data that the BatchManager will submit.
            -'dense' : Simple Dense representation of the data. Ie: ('Hi','yo') => '[[13,7],[24,10]]' => [batch_size x seq_length]
            -'dense_rnn' : Dense representation that matchs TFs RNN Cells expected shape. Ie: ('Hi','yo') => '[[[13],[7]],[[24],[10]]]' => [batch_size x seq_length x num_features]
            -'one_hot' : OneHot representation that matchs TFs RNN Cells expected shape.
        '''

        self._configured = True

        #Read corpus
        if(filename is not None):
            self._filename = filename
            corpus_raw = u""
            with codecs.open(filename,'r','utf-8') as file:
                corpus_raw += file.read()
                self._text = corpus_raw
                self._text_size = len(corpus_raw)

        #Create vocabulary
        if(vocab_strategy is not None):
            self._vocab_strategy = vocab_strategy
            self._create_vocab()

        if(batch_size is not None):
            self._batch_size = batch_size
        if(seq_length is not None):
            self._seq_length = seq_length
        if(overlap is not None):
            self._overlap = overlap
       
        if(mode is not None):
            self._mode = mode

        #Compute num batches
        if(self._batch_size is not None and self._seq_length is not None and self._overlap is not None):
            self._get_num_batches()


    def get_params(self):
        '''
        Prints the current configuration of BatchManager. Also checks whether or not the BatchManager is configured.
        '''

        if(self._configured == False):
            warnings.warn("BatchManager not configured - Halting. Please, set up the mandatory params with BatchManager.set_params()", UserWarning)
        else:
            print('@params:')
            print('[filename : "%s"]' % self._filename)
            print('[text : "%s"]' % self._text[0:10])
            print('[text_size : %s]' % self._text_size)
            print('[vocab : %s]' % self._vocab[0:5])
            print('[vocab_size : %s]' % self._vocab_size)
            print('[mode : %s]' % self._mode)
            print('[batch_size : %s]' % self._batch_size)
            print('[seq_length : %s]' % self._seq_length)
            print('[overlap : %s]' % self._overlap)
            print('[num_batches : %s]' % self._num_batches)



    def _create_vocab(self):
        '''
        Private function. Creates the vocabulary according to the configuration.
        '''

        if(self._vocab_strategy == 'all'):
            vocab = list(set(self._text))
        elif(self._vocab_strategy == 'az'):
            corpus_raw = self._text.lower()
            vocab = re.sub("[^a-z]"," ",corpus_raw)
        elif(self._vocab_strategy == 'az_num'):
            corpus_raw = self._text.lower()
            vocab = re.sub("[^a-z0-9]"," ",corpus_raw)


        self._vocab = vocab
        self._vocab_size = len(vocab)

    def _read_data(self):
        '''
        Private function. Yields data to the batch reader.
        '''

        text = self._text

        #The vocab encoding is done in dense mode despite of the configured mode. The transformation to OneHot, or other modes, if required, are done lately.
        #This could be reviewed.
        text = self.vocab_encode(text,mode='dense')

        #Fragments data into batches and yields them to batch reader
        for start in range(0, self._text_size - self._seq_length, self._overlap):

            xData = text[start: start + self._seq_length]
            xData += [0] * (self._seq_length - len(xData))

            yData = text[start+1: start + self._seq_length+1]
            yData += [0] * (self._seq_length - len(yData))

            yield xData,yData
    
    def _read_batch(self,stream):
        '''
        Private function. Yields the batches. The 'user-level' function to retrieve is not this, instead use BatchManager.generate_batches()
        This function is in charge of making the potentially required transformation from 'dense' encoding to another encoding structure.
        '''

        #Yields the batches in dense mode
        if(self._mode == 'dense'):

            batchX = []
            batchY = []
            for elemX,elemY in stream:
                batchX.append(elemX)
                batchY.append(elemY)
                if len(batchX) == self._batch_size:
                    yield batchX,batchY
                    batchX = []
                    batchY = []
            
        #Yields the batches in dense_rnn mode
        elif(self._mode == 'dense_rnn'):
            batchX = []
            batchY = []
            for elemX,elemY in stream:
                batchX.append(elemX)
                batchY.append(elemY)
                if len(batchX) == self._batch_size:
                    batchX = np.array(batchX)
                    batchX = batchX.reshape(batchX.shape[0],batchX.shape[1],1)
                    batchY = np.array(batchY)
                    batchY = batchY.reshape(batchY.shape[0],batchY.shape[1],1)
                    yield batchX,batchY
                    batchX = []
                    batchY = []
            
        #Yields the batches in one hot mode
        elif(self._mode == 'onehot'):

            batchX = []
            batchY = []
            for elemX,elemY in stream:
                elemOneHotX = self._id_to_one_hot(elemX)
                elemOneHotY = self._id_to_one_hot(elemY)
                batchX.append(elemOneHotX)
                batchY.append(elemOneHotY)
                if len(batchX) == self._batch_size:
                    yield batchX,batchY
                    batchX = []
                    batchY = []
            



    def generate_batches(self,num_epochs=1):
        '''
        Yields batches from the specified file with the correspondent configuration. Each epoch is a complete loop over the full text. 
        args:
        num_epochs: Number of epochs. In other words, full loops over the entire corpus to be done.
        yields:
        batchX : The batch, according to the configuration, meant to be the input data
        batchY : The batch, according to the configuration, meant to be the input labels
        epoch : The current epoch
        Example of use: 
        for batchX,batchY,epoch in BatchManager.generate_batches():
            #Here you get in each iteration the new batches
            ...
        '''

        if(self._configured == False):
            warnings.warn("BatchManager not configured - Halting. Please, set up the mandatory params with BatchManager.set_params()", UserWarning)
            return False

        for epoch in range(num_epochs):
            for batchX,batchY in self._read_batch(self._read_data()):
                yield batchX,batchY,epoch


    def vocab_encode(self,text,mode='set'):
        '''
        Encodes a text. If mode param not specified, the configured mode is used.
        args:
        text : The text to be encoded as 'str'
        mode : The mode to be encoded in ['dense','dense_rnn','onehot']
        returns: The encoded text
        '''

        if(mode == 'set'):
            mode = self._mode

        if(mode == 'dense'):
            return [self._vocab.index(x) for x in text if x in self._vocab]

        elif(mode == 'dense_rnn'):
            return [self._vocab.index(x) for x in text if x in self._vocab]

        elif(mode == 'onehot'):
            encoded_one_hot = np.zeros((len(text),self._vocab_size))
            idxs = [self._vocab.index(x) for x in text if x in self._vocab]
            for i,idx in enumerate(idxs):
                character_one_hot = np.zeros((self._vocab_size))
                character_one_hot[idx] = 1
                encoded_one_hot[i] = character_one_hot


            return encoded_one_hot


    def vocab_decode(self,encoded_text,mode='set'):
        '''
        Decodes a text. If mode param not specified, the configured mode is used.
        args:
        encoded_text : The encoded text in 'dense', 'dense_rnn' or 'onehot' format.
        mode : The mode to be decoded from ['dense','dense_rnn','onehot']
        returns: The decoded text
        '''

        if(mode == 'set'):
            mode = self._mode

        if(mode == 'dense'):
            return ''.join([self._vocab[x] for x in encoded_text])

        elif(mode == 'dense_rnn'):
            real_ids = encoded_text.reshape(-1)
            #Like dense
            return ''.join([self._vocab[x] for x in encoded_text])


        elif(mode == 'onehot'):
            #Iterate the sequence
            #For each one hot vector extract the id where the 1 is and do regular vocab decode
            real_ids = []
            for one_hot_vector in encoded_text:
                id = np.argmax(one_hot_vector)
                real_ids.append(id)

            #Like dense
            return ''.join([self._vocab[x] for x in real_ids])

        

    def _id_to_one_hot(self,list_ids):
        '''Private function. Turns a dense representation of chars to a one hot.
        returns: The One Hot representation of the Dense reppresented characters received by param.
        '''

        elemOneHot = np.zeros(shape=(self._seq_length,self._vocab_size))
        for i,id in enumerate(list_ids):
            elemOneHot[i,id] = 1

        return elemOneHot


    def stats(self):
        '''
        Prints some stats about the batches that will be created according to the current configuration. Also checks whether or not the BatchManager is configured.
        '''
        if(self._configured == False):
            warnings.warn("BatchManager not configured - Halting. Please, set up the mandatory params with BatchManager.set_params()", UserWarning)


        else:
            print('The corpus has %d characters' %self._text_size)
            print('Configuration:\n[batch_size : %d]\n[seq_length : %d]\n[overlap : %d]\n' % (self._batch_size, self._seq_length, self._overlap))

            print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past and overlapping %d steps' 
                %(self._get_num_batches(),self._batch_size,self._seq_length, (self._seq_length-self._overlap)))


    def _get_num_batches(self):
        '''
        Private auxiliary function that calculates how many batches will be created with the current configuration.
        returns:
        num_batches : The number of batches
        '''
        self._num_batches = (self._text_size-1) // (self._batch_size * (self._seq_length - (self._seq_length-self._overlap)))
        return self._num_batches

#################################################END-BATCHMANAGER#################################################

            


