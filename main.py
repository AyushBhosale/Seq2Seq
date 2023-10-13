lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
convers=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')
exchn = []
for conver in convers:
    exchn.append(conver.split(' +++$+++ ')[-1][1:-1].replace("'", " ").replace(",", "").split())

diag = {}
for line in lines:
    diag[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

## delete
del (lines, convers, conver, line)

questions = []
answers = []

for conver in exchn:
    for i in range(len(conver) - 1):
        questions.append(diag[conver[i]])
        answers.append(diag[conver[i + 1]])

## delete
del (diag, exchn, conver, i)

sorted_ques = []
sorted_ans = []
for i in range(len(questions)):
    if len(questions[i]) < 13:
        sorted_ques.append(questions[i])
        sorted_ans.append(answers[i])


import re
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt


clean_ques = []
clean_ans = []

for line in sorted_ques:
    clean_ques.append(clean_text(line))

for line in sorted_ans:
    clean_ans.append(clean_text(line))

for i in range(len(clean_ans)):
    clean_ans[i] = ' '.join(clean_ans[i].split()[:11])

del(sorted_ans, sorted_ques)

clean_ans=clean_ans[:30000]
clean_ques=clean_ques[:30000]

word2count = {}

for line in clean_ques:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for line in clean_ans:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

## delete
del(word, line)
# here we remove less frequent words
thresh = 5
vocab = {}
word_num = 0
for word, count in word2count.items():
    if count >= thresh:
        vocab[word] = word_num
        word_num += 1
## delete
del (word2count, word, count, thresh)
del (word_num)
for i in range(len(clean_ans)):
    clean_ans[i] = '<SOS> ' + clean_ans[i] + ' <EOS>'

tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

vocab['cameron']=vocab['<PAD>']
vocab['cameron']=0
del(token, tokens)
del(x)
inv_vocab={w:v for v,w in vocab.items()}
del(i)

encoder_inp = []
for line in clean_ques:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])

    encoder_inp.append(lst)

decoder_inp = []
for line in clean_ans:
    lst = []
    for word in line.split():
        if word not in vocab:
            lst.append(vocab['<OUT>'])
        else:
            lst.append(vocab[word])
    decoder_inp.append(lst)

### delete
del (clean_ans, clean_ques, line, lst, word)
# cleaning
from keras.preprocessing.sequence import pad_sequences
encoder_inp=pad_sequences(encoder_inp,13,padding='post',truncating='post')
decoder_inp=pad_sequences(decoder_inp,13,padding='post',truncating='post')
decoder_final_output = []
for i in decoder_inp:
    decoder_final_output.append(i[1:])
decoder_final_output = pad_sequences(decoder_final_output, 13, padding='post', truncating='post')
del(i)
print(encoder_inp.shape)
print(decoder_inp.shape)
print(decoder_final_output.shape)
import tensorflow as tf
enc_inp=tf.keras.layers.Input(shape=(13,))
dec_inp=tf.keras.layers.Input(shape=(13,))
VOCAB_SIZE=len(vocab)
embed=tf.keras.layers.Embedding(VOCAB_SIZE+1,output_dim=50,input_length=13,trainable=True)
enc_embeb=embed(enc_inp)
enc_lstm=tf.keras.layers.LSTM(400,return_state=True,return_sequences=True)
enc_op, h, c= enc_lstm(enc_embeb)
enc_states=[h,c]

dec_embeb=embed(dec_inp)
dec_lstm=tf.keras.layers.LSTM(400,return_state=True,return_sequences=True)
dec_op, _, _= dec_lstm(dec_embeb, initial_state=enc_states)

dense=tf.keras.layers.Dense(VOCAB_SIZE,activation='softmax')
dense_op=dense(dec_op)
model=tf.keras.models.Model([enc_inp,dec_inp],dense_op)
model.compile(loss='categorical_crossentropy',
          metrics=['acc'],
          optimizer='adam')
model.fit([encoder_inp,decoder_inp],[decoder_final_output],epochs=40)