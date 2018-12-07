# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # to ignore warnings
import tensorflow as tf
import numpy as np
import time
import re

print ("\n** Welcome to Friendroid ** \n")

min_line_length = 2
max_line_length = 6
threshold = 10

vocab = {}
questions = []
answers = []
clean_questions = []
clean_answers = []
short_questions_temp = []
short_answers_temp = []
short_questions = []
short_answers = []

codes = ['<PAD>','<EOS>','<UNK>','<GO>']

questions_vocab_to_int = {}
answers_vocab_to_int = {}
questions_int_to_vocab = {}
answers_int_to_vocab = {}
questions_int = []
answers_int = []

print ("Reading files from dataset and preprocessing.... \n")
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read()
lines = lines.split('\n')

line_id_to_text = {} # Line no to line text mapping

for row in lines:
	string = row.split(' +++$+++ ')
	if(len(string)==5):
		key = str(string[0]).strip()
		val = str(string[4]).strip()
		val=val.lower()
		line_id_to_text[key]=val

conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read()
conv_lines = conv_lines.split('\n')[:-1]

conversation_flow = [ ]

for line in conv_lines:
	string = line.split(' +++$+++ ')
	string = string[-1]
	string = string[1:-1].replace("'","").replace(" ","") 
	string = string.split(',')
	conversation_flow.append(string)

for conv in conversation_flow:
	for i in range(0, len(conv)-1):
		ques = line_id_to_text[conv[i]]
		questions.append(ques)
		ans = line_id_to_text[conv[i+1]]
		answers.append(ans)

#print ("No of conversation pairs in dataset : "+ str(len(answers)))

def format_text(dialog):
	dialog = re.sub(r"[-()\"'#/@;:<>{}`+=~|.!?,]", "", dialog)
	dialog = dialog.replace("won't", "will not")
	dialog = dialog.replace("can't", "cannot")
	dialog = dialog.replace("n'", "ng")
	dialog = dialog.replace("'bout", "about")
	dialog = dialog.replace("she's", "she is")
	dialog = dialog.replace("it's", "it is")
	dialog = dialog.replace("\'ve", " have")
	dialog = dialog.replace("\'re", " are")
	dialog = dialog.replace("\'d", " would")
	dialog = dialog.replace("that's", "that is")
	dialog = dialog.replace("what's", "what is")
	dialog = dialog.replace("where's", "where is")
	dialog = dialog.replace("how's", "how is")
	dialog = dialog.replace("\'ll", " will")
	dialog = dialog.replace("'til", "until")
	dialog = dialog.replace("i'm", "i am")
	dialog = dialog.replace("he's", "he is")
	dialog = dialog.replace(r"n't", " not")
	return dialog

for answer in answers:
    clean_answers.append(format_text(answer))

for question in questions:
    clean_questions.append(format_text(question))

for i in range(0, len(clean_questions)):
	q = clean_questions[i].split()
	if(len(q)<=max_line_length):
		if(len(q)>=min_line_length):
			short_questions_temp.append(clean_questions[i])
			short_answers_temp.append(clean_answers[i])

for i in range(0, len(short_answers_temp)):
	a = short_answers_temp[i].split()
	if(len(a)<=max_line_length):
		if(len(a)>=min_line_length):
			short_answers.append(short_answers_temp[i])
			short_questions.append(short_questions_temp[i])

print("\nNo of questions : ", len(short_questions))
print("No of answers : ", len(short_answers))

for i in range(0, len(short_questions)):
	question = short_questions[i]
	words = question.split()
	for word in words:
		if word in vocab:
			vocab[word] += 1
		else:
			vocab[word] = 1

for i in range(0, len(short_answers)):
	ans = short_answers[i]
	words = ans.split()
	for word in words:
		if word in vocab:
			vocab[word] += 1
		else:
			vocab[word] = 1
           
word_id = 0
for word in vocab:
	if(vocab[word]>=threshold):
		questions_vocab_to_int[word] = word_id
		answers_vocab_to_int[word] = word_id
		word_id+=1

for code in codes:
	answers_vocab_to_int[code] = len(answers_vocab_to_int)+1
	questions_vocab_to_int[code] = len(questions_vocab_to_int)+1

for i in range(len(short_answers)):
    short_answers[i] += ' <EOS>'
    
for key in questions_vocab_to_int:
	val = questions_vocab_to_int[key]
	questions_int_to_vocab[val] = key

for key in answers_vocab_to_int:
	val = answers_vocab_to_int[key]
	answers_int_to_vocab[val] = key

print ("\nLength of Vocabulary after preprocessing datset : " +str(len(questions_vocab_to_int)))

for i in range(0, len(short_questions)):
	numlist = []
	ques = short_questions[i]
	qlist = ques.split()
	for word in qlist:
		if word in questions_vocab_to_int:
			numlist.append(questions_vocab_to_int[word])
		else:
			numlist.append(questions_vocab_to_int['<UNK>'])
	questions_int.append(numlist)

for i in range(0, len(short_answers)):
	numlist = []
	ans = short_answers[i]
	alist = ans.split()
	for word in alist:
		if word in answers_vocab_to_int:
			numlist.append(answers_vocab_to_int[word])
		else:
			numlist.append(answers_vocab_to_int['<UNK>'])
	answers_int.append(numlist)

print("\nNo of question/answer pairs after preprocessing dataset : "+str(len(questions_int)))

sorted_questions = []
sorted_answers = []
loop_len=max_line_length+1
for length in range(1, loop_len):
    z=enumerate(questions_int)
    for i in z:
        cur_l=len(i[1])
        if  cur_l == length:    
            x=questions_int[i[0]]
            sorted_questions.append(x)
            x=answers_int[i[0]]
            sorted_answers.append(x)

def placeholder_generator():
    '''Create palceholders for inputs to the model'''
    x=[None, None]
    input_data = tf.placeholder(tf.int32,x , name='input')
    x=[None, None]
    targets = tf.placeholder(tf.int32, x, name='targets')
    x='learning_rate'
    lr = tf.placeholder(tf.float32, name=x)
    x='keep_prob'
    keep_prob = tf.placeholder(tf.float32, name=x)
    ret=input_data, targets, lr, keep_prob
    return ret


def input_encoding_processing(target_data, vocab_to_int, batch_size):    
    x=[0,0]
    ending = tf.strided_slice(target_data, x, [batch_size, -1], [1, 1])
    x=[batch_size, 1]
    dec_input = tf.concat([tf.fill(x, vocab_to_int['<GO>']), ending], 1)
    ret=dec_input
    return ret


def layer_encoding(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):    
    x=rnn_size
    lstm = tf.contrib.rnn.BasicLSTMCell(x)
    x=keep_prob
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = x)
    x=num_layers    
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * x)
    inp=[enc_cell,sequence_length,rnn_inputs,tf.float32]
    z, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = inp[0],cell_bw = inp[0],sequence_length = inp[1],inputs = inp[2], dtype=inp[3])
    ret=enc_state
    return ret


def train_decode_layer(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,output_fn, keep_prob, batch_size):
    bsize=batch_size
    attention_states = tf.zeros([bsize, 1, dec_cell.output_size])
    attent_opt="bahdanau"
    att_keys, att_vals, att_score_fn, att_construct_fn =tf.contrib.seq2seq.prepare_attention(attention_states,attention_option=attent_opt,num_units=dec_cell.output_size)    
    name_value="attn_dec_train"
    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],att_keys,att_vals,att_score_fn,att_construct_fn,name = name_value)
    inp=[dec_cell,train_decoder_fn,dec_embed_input, sequence_length,decoding_scope]
    train_pred, z, y = tf.contrib.seq2seq.dynamic_rnn_decoder(inp[0], inp[1], inp[2], inp[3],scope=inp[4])
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    ret=output_fn(train_pred_drop)
    return ret
    x=1


def infer_decode_layer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):        
    bsize=batch_size
    attention_states = tf.zeros([bsize, 1, dec_cell.output_size])
    attent_opt="bahdanau"
    att_keys, att_vals, att_score_fn, att_construct_fn =tf.contrib.seq2seq.prepare_attention(attention_states,attention_option=attent_opt,num_units=dec_cell.output_size)
    inp=[output_fn, encoder_state[0], att_keys, att_vals, att_score_fn, att_construct_fn, dec_embeddings,start_of_sequence_id, end_of_sequence_id, maximum_length, vocab_size,"attn_dec_inf"]
    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(inp[0],inp[1],inp[2],inp[3],inp[4],inp[5],inp[6],inp[7],inp[8],inp[9],inp[10], name = inp[11])
    inp=[dec_cell, infer_decoder_fn,decoding_scope]
    infer_logits, y, z = tf.contrib.seq2seq.dynamic_rnn_decoder(inp[0],inp[1], scope=inp[2])
    ret=infer_logits
    return ret
    x=1


def layer_decoding(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,num_layers, vocab_to_int, keep_prob, batch_size):    
    with tf.variable_scope("decoding") as decoding_scope:
        rsize=rnn_size
        lstm = tf.contrib.rnn.BasicLSTMCell(rsize)
        inp=[lstm,keep_prob]
        drop = tf.contrib.rnn.DropoutWrapper(inp[0], input_keep_prob = inp[1])
        nlayer=num_layers
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * nlayer)
        y=0.1
        weights = tf.truncated_normal_initializer(stddev=y)
        biases = tf.zeros_initializer()
        z=None
        w=weights
        output_fn = lambda x: tf.contrib.layers.fully_connected(x, vocab_size,z, scope=decoding_scope,weights_initializer = w,biases_initializer = biases)      
        bsize=batch_size
        train_logits = train_decode_layer(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope, output_fn, keep_prob, bsize)
        decoding_scope.reuse_variables()
        bsize=batch_size
        seqlen= sequence_length-1
        infer_logits = infer_decode_layer(encoder_state, dec_cell, dec_embeddings, vocab_to_int['<GO>'],vocab_to_int['<EOS>'], seqlen, vocab_size,decoding_scope, output_fn, keep_prob, bsize)
    return train_logits, infer_logits
    x=1



def seqtwoseqmodel(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers, questions_vocab_to_int):    
    rout=tf.random_uniform_initializer(0,1)
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, answers_vocab_size+1, enc_embedding_size,initializer = rout)
    nlayer=num_layers
    enc_state = layer_encoding(enc_embed_input, rnn_size, nlayer, keep_prob, sequence_length)
    bsize=batch_size
    dec_input = input_encoding_processing(target_data, questions_vocab_to_int, bsize)
    rout=tf.random_uniform([questions_vocab_size+1, dec_embedding_size], 0, 1)
    dec_embeddings = tf.Variable(rout)
    input_dec=dec_input
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, input_dec)
    input_embed_dec=dec_embed_input
    train_logits, infer_logits = layer_decoding(input_embed_dec, dec_embeddings, enc_state, questions_vocab_size, sequence_length, rnn_size, num_layers, questions_vocab_to_int, keep_prob, batch_size)
    ret=train_logits, infer_logits
    return ret
    x=1


# In[38]:
def value_assign(val):
    if val==1:
        return 1#epochs
    if val==2:
        return 128#batch size
    if val==3:
        return 512#rnn size
    if val==4:
        return 2#layers
    if val==5:
        return 128#encoding embedding size
    if val==6:
        return 128#decoding embeddin size
    if val==7:
        return .0005# learning rate
    if val==8:
        return .9#epochs
    if val==9:
        return .0001#epochs
    if val==10:
        return .75#keep_prob

# Set the Hyperparameters
epochs = value_assign(1)
batch_size = value_assign(2)
rnn_size = value_assign(3)
num_layers = value_assign(4)
encoding_embedding_size = value_assign(5)
decoding_embedding_size = value_assign(6)
learning_rate = value_assign(7)
learning_rate_decay = value_assign(8)
min_learning_rate = value_assign(9)
keep_probability = value_assign(10)


print ("\n** Training RNN begins... ** \n")
# Reset the graph to ensure that it is ready for training
tf.reset_default_graph()
# Start the session
sess = tf.InteractiveSession()
    
# Load the model inputs    
input_values = placeholder_generator()
input_data=input_values[0]
targets=input_values[1]
lr=input_values[2]
keep_prob=input_values[3]

# Sequence length will be the max line length for each batch
z=None
sequence_length = tf.placeholder_with_default(max_line_length, z, name='sequence_length')
# Find the shape of the input data for sequence_loss

input_shape = tf.shape(input_data)

# Create the training and inference logits
revout=tf.reverse(input_data, [-1])
logits = seqtwoseqmodel(revout, targets, keep_prob, batch_size, sequence_length, len(answers_vocab_to_int), len(questions_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, questions_vocab_to_int)
train_logits=logits[0]
inference_logits=logits[1]
# Create a tensor for the inference logits, needed if loading a checkpoint version of the model
l='logits'
tf.identity(inference_logits, l)

def pad_sentence_batch(sentence_batch, vocab_to_int):
    max_sentence=0
    for sentence in sentence_batch:
        max_sentence = max(len(sentence),max_sentence)
    ret=[sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]
    return ret

with tf.name_scope("optimization"):
    ones=tf.ones([input_shape[0], sequence_length])    
    cost = tf.contrib.seq2seq.sequence_loss(train_logits,targets,ones)
    lrate=learning_rate
    optimizer = tf.train.AdamOptimizer(lrate)   
    gradients = optimizer.compute_gradients(tf.contrib.seq2seq.sequence_loss(train_logits,targets,ones))
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

def batch_data(questions, answers, batch_size):
    """Batch questions and answers together"""
    rnge=len(questions)//batch_size
    for batch_i in range(0, rnge):
        start_i = batch_i * batch_size
        strt=start_i
        end=start_i + batch_size
        questions_batch = questions[strt:end]
        strt=start_i
        end=start_i + batch_size
        answers_batch = answers[strt:end]
        inp=pad_sentence_batch(questions_batch, questions_vocab_to_int)        
        inp1=pad_sentence_batch(answers_batch, answers_vocab_to_int)        
        yield np.array(inp), np.array(inp1)

train_set=len(sorted_questions)*0.10

# Split the questions and answers into training and validating data
train_questions = sorted_questions[int(train_set):]
train_answers = sorted_answers[int(train_set):]

valid_questions = sorted_questions[:int(train_set)]
valid_answers = sorted_answers[:int(train_set)]


def value_assign_1(val):
    if val==1:
        return 100#display steps
    if val==2:
        return 0#stop early
    if val==3:
        return 5#stop
    if val==4:
        return ((len(train_questions))//batch_size//2)-1#validation check
    if val==5:
        return 0#total train loss
    if val==6:
        return []#summary valid loss

display_step = value_assign_1(1) # Check training loss after every 100 batches
stop_early = value_assign_1(2)
stop = value_assign_1(3) # If the validation loss does decrease in 5 consecutive checks, stop training
validation_check = value_assign_1(4) # Modulus for checking validation loss
total_train_loss = value_assign_1(5) # Record the training loss for each display step
summary_valid_loss = value_assign_1(6) # Record the validation loss for saving improvements in the model

checkpoint = "model.ckpt" 

sess_inp=tf.global_variables_initializer()
sess.run(sess_inp)

print ("\nStarting Epochs... \n")
for epoch_i in range(1, epochs+1):
    enum_out=enumerate(batch_data(train_questions, train_answers, batch_size))
    for batch_i, ques_ans_batch in enum_out:
        start_time = time.time()
        gama, loss = sess.run([train_op, cost],{input_data: ques_ans_batch[0],targets: ques_ans_batch[1],lr: learning_rate,sequence_length: ques_ans_batch[1].shape[1],keep_prob: keep_probability})
        total_train_loss = total_train_loss+loss    
        batch_time = time.time() - start_time

        if batch_i % display_step == 0:
            deno = len(train_questions) // batch_size
            print('Epoch '+str(epoch_i)+"/"+str(epochs)+ " Batch " + str(batch_i)+"/"+str(deno)+" - Loss: "+str(total_train_loss / display_step)+" Seconds: "+str(batch_time*display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            enum_out=enumerate(batch_data(valid_questions, valid_answers, batch_size))
            for batch_ii, ques_ans_batch in enum_out:
                valid_loss = sess.run(cost, {input_data:ques_ans_batch[0],targets: ques_ans_batch[1],lr: learning_rate,sequence_length: ques_ans_batch[1].shape[1],keep_prob: 1})
                total_valid_loss += valid_loss          
            batch_time = time.time() - start_time
            denom=(len(valid_questions) / batch_size)
            avg_valid_loss = total_valid_loss / denom
            print("Valid Loss: "+str(avg_valid_loss) +" Seconds: "+str(batch_time))
            learning_rate =learning_rate*learning_rate_decay
            learning_rate = min(learning_rate,min_learning_rate)

            summary_valid_loss.append(avg_valid_loss)
            flag=avg_valid_loss <= min(summary_valid_loss)
            if flag:
                msg="New record"
                print(msg) 
                stop_early = 0
                saver = tf.train.Saver() 
                saver.save(sess, checkpoint)
            if not(flag):
                msg="No Improvement"
                print(msg)
                stop_early = stop_early+1
                flag1=stop_early == stop
                if flag1:
                    break
    flag2=  stop_early == stop
    '''
    if flag2:
        msg="Stopping Training."
        print(msg)
        break
    '''
