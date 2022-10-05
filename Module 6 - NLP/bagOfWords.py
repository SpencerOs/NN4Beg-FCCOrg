vocab = {} # a dictionary of word -> int representation
word_encoding = 1
def bag_of_words(text):
    global word_encoding

    words = text.lower().split(" ") # create a list of all of the words in the text, we'll assume there is no grammar in our text for this example
    bag = {} # stores all of the encodings and their frequency

    for word in words:
        if word in vocab:
            encoding = vocab[word]
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1
        
        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
        
    return bag
text = "This is a test to see if this test will work is is test a a"
bag = bag_of_words(text)
print(bag)
print(vocab)

positive_review = "I thought the movie was going to be bad but it was actually amazing"
negative_review = "I thought the movie was going to be amazing but it was actually bad"

pos_bag = bag_of_words(positive_review)
neg_bag = bag_of_words(negative_review)

print("Positive: {}".format(pos_bag))
print("Negative: {}".format(neg_bag))

# ------------------------------------------------------------------------------------
# Integer Encoding
# ------------------------------------------------------------------------------------
vocab = {}
word_encoding = 1
def one_hot_encoding(text):
    global word_encoding

    words = text.lower().split(" ")
    encoding = []

    for word in words:
        if word in vocab:
            code = vocab[word]
            encoding.append(code)
        else:
            vocab[word] = word_encoding
            encoding.append(word_encoding)
            word_encoding += 1
    return encoding

encoding = one_hot_encoding(text)
print("\n\n--------------------------------------------------------------\nInteger Encoding\n{}".format(encoding))
print(vocab)

pos_encode = one_hot_encoding(positive_review)
neg_encode = one_hot_encoding(negative_review)
print("Positive: {}".format(pos_encode))
print("Negative: {}".format(neg_encode))