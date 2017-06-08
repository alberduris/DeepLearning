
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


# In[3]:

corpus = read_data('got1.txt')
print('Data sample (First 140 characters):\n%s ...' %corpus[0:140])


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


# In[5]:

filenames = ['got1.txt','tiny_shakespeare.txt']
corpus = read_and_merge_files(filenames)
print('Data sample (First and last 140 characters):\n %s // %s' %(corpus[0:140],corpus[-140:]))


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


# In[7]:

sentences = extract_sentences(corpus=corpus)
print('The first 5 sentences as sample:\n %s' %(sentences[0:5]))


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


# In[9]:

vocab = extract_words(corpus=corpus)
print('The first 10 words as sample:\n %s' %(vocab[0:10]))


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


# In[11]:

characters = extract_characters(corpus)
print('The first 140 characters as sample:\n %s' %(characters[0:25]))


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


# In[13]:

unique_words = extract_unique_words(corpus)
print('Size of vocabulary (words): %d' %len(unique_words))
print('The first 10 words as sample:\n %s' %(unique_words[0:10]))


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


# In[15]:

unique_characters = extract_unique_characters(corpus)
print('Size of vocabulary (characters): %d' %len(unique_characters))
print('The first 10 characters as sample:\n %s' %(unique_characters[0:10]))


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


# With **words**

# In[17]:

import itertools
import collections

id_to_vocab,vocab_to_id = create_dictionaries(vocab=unique_words)

print('The first 4 elements of dict [id_to_vocab] as sample:')
x = itertools.islice(id_to_vocab.items(), 0, 4)
for key, value in x:
    print (key, value)
    
print('\nThe first 4 elements of dict [vocab_to_id] as sample:')
y = itertools.islice(vocab_to_id.items(), 0, 4)
for key, value in y:
    print (key, value)
    


# With **characters**

# In[18]:

import itertools
import collections

id_to_char,char_to_id = create_dictionaries(unique_characters)

print('The first 4 elements of dict [id_to_char] as sample:')
x = itertools.islice(id_to_char.items(), 0, 4)
for key, value in x:
    print (key, value)
    
print('\nThe first 4 elements of dict [char_to_id] as sample:')
y = itertools.islice(char_to_id.items(), 0, 4)
for key, value in y:
    print (key, value)


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


# With **words**

# In[20]:

corpus_by_words_as_int = extract_corpus_as_integers(corpus,vocab_to_id)
print('Same text: as integers, as words:')
print(corpus_by_words_as_int[0:10])
list_words_as_words = []
for ix in corpus_by_words_as_int[0:10]:
    list_words_as_words.append(id_to_vocab[ix])
print(list_words_as_words)


# With **characters**

# In[21]:

corpus_by_chars_as_int = extract_corpus_as_integers(corpus,char_to_id,level='chars')
print('Same text, as integers, as words:')
print(corpus_by_chars_as_int[0:10])
list_chars_as_chars = []
for ix in corpus_by_chars_as_int[0:10]:
    list_chars_as_chars.append(id_to_char[ix])
print(list_chars_as_chars)


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
    


# In[30]:

#Because the unique words size in the example corpus is ~15.000, with a vocabulary_size less than that, we'll have unknown words
data,count,dictionary,reverse_dictionary = build_dataset(extract_words(corpus),vocabulary_size=10000)

print('The first 4 elements of list that holds the counting [count] as sample:\n %s\n' % count[0:4])

print('The first 4 elements of dict [dictionary] as sample:')
x = itertools.islice(dictionary.items(), 0, 4)
for key, value in x:
    print (key, value)
    
print('\nThe first 4 elements of dict [reverse_dictionary] as sample:')
x = itertools.islice(reverse_dictionary.items(), 0, 4)
for key, value in x:
    print (key, value)

print('\nSame text: as integers, as words:')
print(data[0:10])
list_words_as_words = []
for ix in data[0:10]:
    list_words_as_words.append(reverse_dictionary[ix])
print(list_words_as_words)



# # Generate Word2Vec Batches

# In[31]:

def generate_batch(data,batch_size=16,num_skips=1,skip_window=1):
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


# With **words**:

# In[34]:

batch,labels = generate_batch(data=corpus_by_words_as_int,batch_size=16,num_skips=1,skip_window=1)


# In[35]:

print('The generated batch is:\n%s\n' %batch)
print('Translated to words are:')
translation = []
for ix in batch:
    translation.append(id_to_vocab[ix])
print(translation)

print('\nThe generated labels for the batch are:\n%s\n' %labels.reshape(-1))
print('Translated to words are:')
translation = []
for ix in labels.reshape(-1):
    translation.append(id_to_vocab[ix])
print(translation)


# With **characters**

# In[36]:

batch,labels = generate_batch(data=corpus_by_chars_as_int,batch_size=16,num_skips=1,skip_window=1)


# In[38]:

print('The generated batch is:\n%s\n' %batch)
print('Translated to characters are:')
translation = []
for ix in batch:
    translation.append(id_to_char[ix])
print(translation)

print('\nThe generated labels for the batch are:\n%s\n' %labels.reshape(-1))
print('Translated to words are:')
translation = []
for ix in labels.reshape(-1):
    translation.append(id_to_char[ix])
print(translation)


# In[ ]:



