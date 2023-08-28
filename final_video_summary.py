import numpy as np
import gc
import os,sys
import glob
# specify the gpu requirements
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "2"
session = tf.Session(config=config)
import cv2
import skimage
from tensorflow.python.keras import backend as K
K.set_session(session)
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras import backend as K
import pandas as pd
import os
import time
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.logging.set_verbosity(tf.logging.ERROR)
import math
import re
import timeit
from collections import Counter
from flask import Flask, render_template, jsonify, request
from flask import g
import cv2
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from summarizer import transformer5
import json
import language_check
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

WORD = re.compile(r"\w+")

app = Flask(__name__)

def video_split():
    required_video_file = "./Input_Video/output.avi"
    output_video_path = "./Test_Videos/"

    #with open("times.txt") as f:
    #  times = f.readlines()

    #times = [x.strip() for x in times] 
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(required_video_file)
    time = clip.duration 
    starttime = 0
    endtime = 10
    for i in range(20):
      #starttime = int(time.split("-")[0])
      #endtime = int(time.split("-")[1])
      ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname= output_video_path + "output_"+str(starttime)+"_"+str(endtime)+".avi")
      starttime = int(starttime)
      endtime = int(endtime)
      starttime = starttime + 10
      endtime = endtime + 10
      if endtime >= time:
        break    

# we need to preprocess the frames in the video before feeding them into the cnn
def preprocess_frames(image, req_width = 224, req_height = 224):
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None],3)
    if len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape

    if width==height:
        resized_image = cv2.resize(image, (req_height,req_width))

    elif height < width:

        # first scale out the image to height = (w/h)*req_height and width = req_width
        resized_image = cv2.resize(image, (int(width * float(req_height) / height), req_width))

        # now find the cropping length
        # since cropping is done from both the ends so cropping length is divided by 2
        cropping_length = int((resized_image.shape[1] - req_height) / 2)

        # crop the image's height from top and bottom
        resized_image = resized_image[:, cropping_length:resized_image.shape[1] - cropping_length]

    else:

        # first scale the image to height = req_height and width = (h/w)*req_width
        resized_image = cv2.resize(image, (req_height, int(height * float(req_width) / width)))

        # now find the cropping length
        # since cropping is done from both the ends so cropping length is divided by 2
        cropping_length = int((resized_image.shape[0] - req_width) / 2)

        # crop the images width from left and right
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length, :]

    return cv2.resize(resized_image, (req_height, req_width))

def extract_features(img , model):
    img = preprocess_input(img)    
    #print "after preprocessing"
    #print img
    #print img.shape
    features = model.predict(img)
    #print "after model.predict"
    #print features.shape
    #print features
    features = np.array(list(map(lambda x: np.squeeze(x), features)))
    #np.squeeze(features)
    #print "After squeeze"
    #print features.shape
    #print features
    return features

def preprocess():
    #vid = int(sys.argv[1]) 
    frames_to_sample = 30
    print ("frames to sample from each video = "+str(frames_to_sample))
    video_dir = './Test_Videos'
    video_save_dir = './Test_Features' 
    videos = os.listdir(video_dir)
    videos = list(filter(lambda x:x.endswith('avi'),videos))
    #videos = video
    #print(video)
    # load vgg16 model pretrained on imagenet dataset
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    for index,video in enumerate(videos):
        video = videos[index]
        print(str(index)+" -----> "+str(video))
        index += 1
        if os.path.exists(os.path.join(video_save_dir, video + '.npy')):
            continue
        
        video_path = os.path.join(video_dir, video)
        try:
            cap = cv2.VideoCapture( video_path )
        except:
            pass

        frame_count = 0
        frame_list = []

        while True:
            ret, frame = cap.read()
            if ret is False:
                break
            frame_list.append(frame)
            frame_count += 1

        frame_list  = np.array(frame_list)

        # if frame count is more than 80 then extract out 80 frames using a linear spacing
        if frame_count > 30:
            frame_indices = np.linspace(0, frame_count, num=frames_to_sample, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        # before extracting the features from the frames we need to preprocess the frames
        #cv2.imwrite('b4.png',frame_list[70])
        cropped_frame_list = np.array(list(map(lambda x: preprocess_frames(x), frame_list)))

	    #print cropped_frame_list.shape
        #cropped_frame_list = np.array(frame_list)
        print(cropped_frame_list.shape)
        cropped_frame_list *= 255 
        feats = np.zeros([len(cropped_frame_list)]+[4096])
        batch_size = 30
        for start,end in zip(range(0,len(cropped_frame_list)+batch_size,batch_size),range(batch_size,len(cropped_frame_list)+batch_size,batch_size)):
            feats[start:end] = extract_features(cropped_frame_list[start:end],model)
            #print(feats)
            #print(feats.shape)
	    # feats has dimension of (number of frames, 4096)

       	save_full_path = os.path.join(video_save_dir, video + '.npy')
        np.save(save_full_path, feats)
       

class V2D():
    def __init__(self, dim_image, vocabulary, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.dim_image = dim_image      # dimension of the image feature extracted from the cnn
        self.vocabulary = vocabulary    # number of words
        self.dim_hidden = dim_hidden    # memory state of lstm, also dimension of the words in vocabulary
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step

        #with tf.device("/gpu:2"):
            # for every word we need to have a word embedding of 1000 dimensions (dim-hidden)
        #if isTrain:
        self.Wemb = tf.Variable(tf.random_uniform([vocabulary, dim_hidden], -0.1, 0.1), name='Wemb')
        #else:  
    #    self.Wemb = tf.get_variable('Wemb', shape = [vocabulary, dim_hidden])
        # create 2 LSTM cells with 1000 hidden units
        # state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state. If False,
        # they are concatenated along the column axis.
        self.lstm1 = tf.nn.rnn_cell.LSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.LSTMCell(dim_hidden, state_is_tuple=False)

        # encode the 4096 dimensional dim_image feature vector
        # for LSTM 1
        #if isTrain:
        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')
    #else:
        #    self.encode_image_W = tf.get_variable(shape = [dim_image, dim_hidden], name="encode_image_W")
    #    self.encode_image_b = tf.get_variable(shape = [dim_hidden], name='encode_image_b')
        #for LSTM 2 we define the weights and biases
        #if isTrain:
        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, vocabulary], -0.1, 0.1), name='embed_word_W')
    #else:
        #    self.embed_word_W = tf.get_variable(shape = [dim_hidden, vocabulary], name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([vocabulary]), name='embed_word_b')

    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        # dimension reshaped to (80, 4096)
        video_flat = tf.reshape(video, [-1, self.dim_image])
         
         # do the matrix multiplication operation and addition of biases
         # encode_image_W has dimension = (4096,1000)
         # encode_image_b has dimension = (1000)
         # video_flat has shape = (80, 4096)    
         # obtained dimension = (80, 1000)
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)

         # reshape back to (1,80,1000)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])
     
         # stores the max probabilty word index in the vocabulary for all the timesteps
         # dimenstion - (20, 1)
        generated_words = []

         # stores the logit words that is the probability of all the words in the vocab for all the timesteps
         # dimension - (20, vocabulary)
        probs = []

         # stores the word embedding of the words for all the timesteps 
         # dimension - (20, 1000)
        embeds = []

        for i in range(0,self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_caption_lstm_step):
            if i == 0:
                #with tf.device('/gpu:2'):
        # find the word embedding for [1] 
        # 1 because index 1 of the vocabulary is <bos>
                current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))
            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            #with tf.device("/gpu:2"):
            current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
            current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds
 

    def build_model(self):

        # for every video in the batch(50), there are n_video_lstm_step(80) represented by a vector of length 1000
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image], name = "video")

        # 1 - for video input and  0 - for no video input 
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step], name = "video_mask")

        #  placeholder that holds the captions
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step + 1], name = "caption")

        # caption word present - 1 not present - 0
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step + 1], name = "caption_mask")

        # flatten the video placeholder shape(50,80,4096) to (4000,4096) shape
        video_flat = tf.reshape(video, [-1, self.dim_image])

        # do the matrix multiplication operation and addition of biases
        # encode_image_W has dimension = (4096,1000)
        # encode_image_b has dimension = (1000)
        # video_flat has shape = (4000, 4096)
        # obtained dimension = (4000, 1000)
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        # reshape from (4000, 1000) back to (50, 80, 1000)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0
        lbls = []
        predictions = []
        # encoding phase
        for i in range(0, self.n_video_lstm_step):
            if i>0:
                tf.get_variable_scope().reuse_variables()

            # get the state (50,2000) and output(50,1000) from the lstm1 and use it over the timestpes
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)
                # As per the paper zeroes are padded to the output of the lstm1 and the fed into the lstm2
                # dimension of output1 = (50, 1000) for ith step
                # dimension of padding = (50, 1000)
                # after concatenation dimension becomes = (50, 2000)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

                # output2 dimension = (50, 1000) for ith step

        # decoding step
        print("---- decoding ----")
        for i in range(0, self.n_caption_lstm_step):
            #with tf.device("/gpu:2"):
                # looks up the embedding for all the words of all the batches for the current lstm step
            tf.get_variable_scope().reuse_variables()

            current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            # for the ith timestep get all the caption placeholders
            # labels = tensor of shape (50,1)
            labels = tf.expand_dims(caption[:, i + 1], 1)
            # generate an indexing from 0 to batchsize-1
            # tf.range(start, limit, delta) just like np.arange()
            # labels = tensor of shape (50,1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)

            # concat both these to get a tensor of shape (50,2)
            # concated stores the complete index where 1 should be placed, on all other places 0s are placed
            concated = tf.concat([indices, labels], 1)

            # onehot encoding for the words - dimension is (50, vocabulary)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.vocabulary]), 1.0, 0.0)

            # logit_words has dimension (50, vocabulary)
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)

            # calculate the cross-entropy loss of the logits with the actual labels
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels)

            # find cross_entropy loss only where mask = 1
            cross_entropy = cross_entropy * caption_mask[:, i]

            # store the probabilities
            probs.append(logit_words)
            lbls.append(onehot_labels)
            current_loss = tf.reduce_sum(cross_entropy) / self.batch_size
            loss = loss + current_loss
            predictions.append(tf.nn.softmax(logit_words)) 
        return loss, video, video_mask, caption, caption_mask, probs, predictions, lbls

model_path = './models/'
video_test_feat_path = './TestFeatures/'

dim_image = 4096
dim_hidden= 1000

n_video_lstm_step = 30
n_caption_lstm_step = 30
n_frame_step = 30

n_epochs = 101
batch_size = 50
learning_rate = 0.0001



def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

def get_video_test_data(video_feat_path):
    '''video_data = pd.read_csv(video_data_path, sep=',')
    #video_data = video_data[video_data['Language'] == 'English']
    video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(video_feat_path, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists( x ))]
    #video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]
    unique_filenames = sorted(video_data['video_path'].unique())
    test_data = video_data[video_data['video_path'].map(lambda x: x in unique_filenames)]'''
    test_data = '/TestFeatures/0lh_UWF9ZP4_30_45.avi.npy'
    return test_data

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))

    word_counts = {}
    nsents = 0
    # iterate over all sentences
    for sent in sentence_iterator:
        nsents += 1
        # iterate over all the words of every sentence to update word counts
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    # make vocabulary of all the words that have word count greater than threshold 
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    # create array for idToWords
    idtoword = {}
    idtoword[0] = '<pad>'
    idtoword[1] = '<bos>'
    idtoword[2] = '<eos>'
    idtoword[3] = '<unk>'

    wordtoid = {}
    wordtoid['<pad>'] = 0
    wordtoid['<bos>'] = 1
    wordtoid['<eos>'] = 2
    wordtoid['<unk>'] = 3

    for idx, w in enumerate(vocab):
        wordtoid[w] = idx+4
        idtoword[idx+4] = w

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    # dimension = size of vocabulary
    bias_init_vector = np.array([1.0 * word_counts[ idtoword[i] ] for i in idtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range

    return wordtoid, idtoword, bias_init_vector

def print_caption(caption, idtoword):
    cp = ""
    for idx in caption:
        cp+=idtoword[idx]+" "
    print(cp)

def test(model_path='./models/'):
    #test_data = get_video_test_data(video_test_feat_path)
    #print(test_data)
    summary = ''
    list_sentences = []
    similarity = 0

    arr = os.listdir('./Test_Features')
    for i in range(len(arr)):
        #print(arr[i])
        test_videos = './Test_Features/' + arr[i] 
        #test_video_names = '_0nX-El-ySo'
        #test_videos = test_data['video_path'].unique()
        #test_video_names = test_data['VideoID'].unique()
        idtoword = pd.Series(np.load('./data/idtoword.npy', allow_pickle = True).tolist())
        tf.reset_default_graph()
        bias_init_vector = np.load('./data/bias_init_vector.npy', allow_pickle = True)
        model = V2D(
            dim_image=dim_image,
            vocabulary=len(idtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            #isTrain = True,
            bias_init_vector=bias_init_vector)
        #for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name):
         #   print i 

        '''config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "3"
        #config.gpu_options.per_process_gpu_memory_fraction=0.90
        config.gpu_options.allow_growth=True
        saver = tf.train.import_meta_graph(model_path)
        with tf.Session(config=config).as_default() as sess:
        #sess.run(tf.global_variables_initializer())        
            #saver = tf.train.import_meta_graph(model_path)
            saver.restore(sess, tf.train.latest_checkpoint('./models1/'))
            print Wemb.eval()'''
        video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()
        saver = tf.train.Saver()

        #sess = tf.InteractiveSession()
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = "1"
        #config.gpu_options.per_process_gpu_memory_fraction=0.90
        config.gpu_options.allow_growth=True
        sess = tf.InteractiveSession(config=config)
        sess.run(tf.global_variables_initializer()) 
        saver.restore(sess, os.path.join(model_path, 'model-100'))
        #print(sess.run(model.Wemb))
        #test_output_txt_fd = open('S2VT_results4.txt', 'w')
        #test_actual_output_txt_fd = open('actual_captions.txt','w')
        #for idx, row in test_data.iterrows():
        #    test_actual_output_txt_fd.write(str(row['VideoID'])+"\t")
        #    test_actual_output_txt_fd.write(str(row['caption'])+"\n")
        #for idx, video_feat_path in enumerate(test_videos):
        #    print(idx, video_feat_path)
            #video_feat_path = '/home/shashwatcs15/VideoAnalytics/La3qC1Y6cX4_9_30.avi.npy'
        #    video_feat = np.load(video_feat_path)[None, ...]
            # video_feat = np.load(video_feat_path)
            # video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

        video_feat = np.load(test_videos, allow_pickle = True)
        video_feat = video_feat.reshape(1,30,4096)
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        #else:
        #    continue
        #video_tf.reshape(80,4096)
        #print(video_mask_tf.shape)
        #print(video_tf.shape)
        #print(video_feat.shape)
        #print(video_mask.shape)
        generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
        #print(sess.run(model.embed_word_W))
        #print(generated_word_index)
        generated_words = idtoword[generated_word_index]

        #punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        #generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        generated_sentence = generated_sentence.replace(' <pad>', '')
        generated_sentence = tool.correct(generated_sentence)
        print(generated_sentence)
        #list_sentences.append(generated_sentence)
        if i == 0:
           #print(generated_sentence)
            list_sentences.append(text_to_vector(generated_sentence))
            summary = summary + generated_sentence + '. '
        for j in range(len(list_sentences)):
            vect_1 = text_to_vector(generated_sentence)
            temp = get_cosine(list_sentences[j], vect_1)
            #print(temp)
            if similarity < temp:
                similarity = temp
        #print(similarity)
        if similarity < 0.9:
            list_sentences.append(text_to_vector(generated_sentence))
            summary = summary + generated_sentence + '. '

            #print(summary)
        similarity = 0
    print(summary)
    tf.get_default_graph()
    return summary
    #test_output_txt_fd.write(test_video_names[idx] + '\t')
    #test_output_txt_fd.write(generated_sentence + '\n')
        #break

import nltk
from nltk.corpus import stopwords
import numpy as np
import networkx as nx

def abstractive_summary(text):
    import torch
    import json 
    from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    device = torch.device('cpu')

    preprocess_text = text.strip().replace("\n","")
    t5_prepared_Text = "summarize: "+preprocess_text
    print ("original text preprocessed: \n", preprocess_text)

    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


    # summmarize 
    summary_ids = model.generate(tokenized_text,
                                        num_beams=4,
                                        no_repeat_ngram_size=2,
                                        min_length=50,
                                        max_length=200,
                                        early_stopping=True)

    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print ("\n\nSummarized text: \n",output)
    return output

# let's begin
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video', methods=['POST','GET'])
def video():
    if request.method == 'GET':
        return render_template('video.html')

    fileInput = request.files['video'] 
    video = fileInput.read()
    FILE_OUTPUT = './Input_Video/output.avi'

    # Checks and deletes the output file
    # You cant have a existing file or it will through an error
    if os.path.isfile(FILE_OUTPUT):
        os.remove(FILE_OUTPUT)

    out_file = open(FILE_OUTPUT, "wb") # open for [w]riting as [b]inary
    out_file.write(video)
    out_file.close()
    #print(type(video))
    #video = str(video.strip())
    start = timeit.default_timer()

    video_split()
    preprocess()
    correct_text = test()
    #correct_text = tool.correct(summary)
    #print(correct_text)

    #final_summary = generate_summary(correct_text, 6)
    #print(final_summary)

    abs_summ = abstractive_summary(correct_text)
    abs_summ = tool.correct(abs_summ)
    stop = timeit.default_timer()
    print(abs_summ)
    path_to_dir  = './Test_Videos'  # path to directory you wish to remove
    files_in_dir = os.listdir(path_to_dir)     # get list of files in the directory

    for file in files_in_dir:                  # loop to delete each file in folder
        os.remove(f'{path_to_dir}/{file}') 
    path_to_dir  = './Test_Features'  # path to directory you wish to remove
    files_in_dir = os.listdir(path_to_dir)     # get list of files in the directory

    for file in files_in_dir:                  # loop to delete each file in folder
        os.remove(f'{path_to_dir}/{file}') 

    print("Timer for video files: ", stop-start)
    return render_template('video.html', score = abs_summ)

import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
import speech_recognition as sr 
  
import os 
  
from pydub import AudioSegment 
from pydub.silence import split_on_silence 
  
# a function that splits the audio file into chunks 
# and applies speech recognition 
from google.cloud import storage
import logging
import os
import cloudstorage as gcs
#import webapp2

#from google.appengine.api import app_identity



def upload_blob():
    """Uploads a file to the bucket."""
    bucket_name = "mfd-audio"
    source_file_name = "output.wav"
    destination_blob_name = "audio"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def audio_summary(channel):
    import io
    import os

    # Imports the Google Cloud client library
    from google.cloud import speech
    #from google.cloud import speech

    # Instantiates a client
    text_speech = ''
    client = speech.SpeechClient()

    # The name of the audio file to transcribe
    #file_name = os.path.join(os.path.dirname(r"C:\Users\Harishankar\Desktop\LibriSpeech\dev-clean\84\121123"),"121123", "84-121123-0001.flac")
    file_name = os.path.join("output.wav")
    #file_name = os.path.join(os.path.dirname(r"C:\Users\Harishankar\Desktop\MP\audio"),"audio", "OSR_us_000_0010_8k.wav")
    #file_name = os.path.join(os.path.dirname(r"C:\Users\Harishankar\Desktop\MediaFileDescriptor"),"MediaFileDescriptor","trial.flac")

    # Loads the audio into memory
    #with io.open(file_name, "rb") as audio_file:
    #    content = audio_file.read()
    #    audio = speech.RecognitionAudio(content=content)
        
    # gcp
    audio = speech.RecognitionAudio(uri="gs://mfd-audio/audio")

    '''config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code="en-US"
    )'''
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        #sample_rate_hertz=16000,
        audio_channel_count = channel,
        language_code="en-US"
    )
      # [START speech_python_migration_async_respons
    operation = client.long_running_recognize(
        request={"config": config, "audio": audio}
        )
    operation = client.long_running_recognize(config=config, audio=audio)

    # Detects speech in the audio file
    response = operation.result(timeout = 90)
    text = []
    for result in response.results:
        #print("Transcript: {}".format(result.alternatives[0].transcript))
        #print("Confidence: {}".format(result.alternatives[0].confidence))
        text.append(result.alternatives[0].transcript)
        text_speech = text_speech + tool.correct(result.alternatives[0].transcript) + '.'
    
    #print(text_speech)
    return text_speech

@app.route('/audio', methods=['POST','GET'])
def audio():
    if request.method == 'GET':
        return render_template('audio.html')

    fileInput = request.files['audio'] 
    video = fileInput.read()
    FILE_OUTPUT = 'output.wav'

    # Checks and deletes the output file
    # You cant have a existing file or it will through an error
    if os.path.isfile(FILE_OUTPUT):
        os.remove(FILE_OUTPUT)

    out_file = open(FILE_OUTPUT, "wb") # open for [w]riting as [b]inary
    out_file.write(video)
    out_file.close()
    start = timeit.default_timer()
    upload_blob()
    channel = 1
    text = audio_summary(channel)
    stop = timeit.default_timer()
    #print(text)
    final_summary = transformer5(text)
    print("Timer for Audio files: ", stop-start)

    return render_template('audio.html', score = final_summary)

@app.route('/video_audio', methods=['POST', 'GET'])
def video_audio_summary():
    import moviepy.editor as mp
    if request.method == 'GET':
        return render_template('video_audio.html')
    fileInput = request.files['video_audio']

    #video to file system
    video = fileInput.read()
    FILE_OUTPUT = './output_vid.mp4'
    if os.path.isfile(FILE_OUTPUT):
        os.remove(FILE_OUTPUT)
    out_file = open(FILE_OUTPUT, "wb") # open for [w]riting as [b]inary
    out_file.write(video)
    out_file.close()
    
    #video to audio and audio to file system
    clip = mp.VideoFileClip("output_vid.mp4") 
    clip.audio.write_audiofile("output.wav") 
    
    #upload and speech to text of audio
    '''file = AudioSegment.from_file("audio.mp3","mp3")
    file.export("audio.flac",format = "flac")
    upload_blob()
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri="gs://mfd-audio/audio")
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        language_code="en-US",
        audio_channel_count = 2
    )
    operation = client.long_running_recognize(
        request={"config": config, "audio": audio}
        )
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout = 300)
    text = ""
    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))
        text = text +" ." + result.alternatives[0].transcript

    abs_summ = abstractive_summary(text)
    abs_summ = tool.correct(abs_summ)'''
    upload_blob()
    channel = 2
    text = audio_summary(channel)
    summ = abstractive_summary(text)
    return render_template('video_audio.html', score = summ)

if __name__ == "__main__":
    app.run(debug = True)

'''preprocess()
correct_text = test()
#correct_text = tool.correct(summary)
print(correct_text)
final_summary = generate_summary(correct_text, 6)
print(final_summary)'''