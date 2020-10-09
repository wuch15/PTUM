#!/usr/bin/env python
# coding: utf-8



from keras.utils.np_utils import to_categorical
from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *
import keras
from keras.layers import *


import pickle
with open('word_dict.pkl','rb')as f:
    word_dict=pickle.load(f)



class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)



with open('/data/wuch/SessionDataDwell/demographicuser.pkl','rb')as f:
    demographicuser=pickle.load( f)

with open('/data/wuch/SessionDataDwell/RawSQUET.tsv','r') as f:
    rawdata=f.readlines()




userdata={}
from nltk.tokenize import wordpunct_tokenize
for i in rawdata:
    tp=i.strip().split('\t')
    if tp[0] not in demographicuser:
        continue
    if tp[0] not in userdata:
        userdata[tp[0]]=[[],[]]
    if tp[2]=='RawSearchQuery':
        userdata[tp[0]][0].append([wordpunct_tokenize(tp[3]),int(tp[5])])
    if tp[2]=='UetPageTitle':
        userdata[tp[0]][1].append([wordpunct_tokenize(tp[3]),int(tp[5])])

del rawdata #save memory


for i in userdata:
    userdata[i][0]=sorted(userdata[i][0],key=lambda x:x[1])
    userdata[i][1]=sorted(userdata[i][1],key=lambda x:x[1])


user_title = []
user_titlepos = []
user_timeinterval = []

MAXT=100
MAXTL=30
user_age_label=[]
user_gender_label=[]
for i in userdata:
    if i not in demographicuser:
        continue
    tt=[]
    for j in userdata[i][1]:
        ttt=[]
        for m in j[0]:
            if m in word_dict :
                ttt.append(word_dict[m][0])

        ttt=ttt[:MAXTL]
        tt.append(ttt+[0]*(MAXTL-len(ttt)))
    tt=tt[:MAXT]
    user_titlepos.append(list(range(1,len(tt)+1))+[0]*(MAXT-len(tt)))
    user_title.append(tt+[[0]*MAXTL]*(MAXT-len(tt)))
    lefttime=[0]+[int(np.log2(userdata[i][1][x+1][1]-userdata[i][1][x][1]+1)) for x in range(len(userdata[i][1][:100])-1)]
    righttime=lefttime[1:]+[0]  
    righttime+=[0]*(100-len(tt))
    user_timeinterval.append(righttime)
    user_age_label.append(demographicuser[i][0])
    user_gender_label.append(demographicuser[i][1])
    



import numpy as np
user_title=np.array(user_title,dtype='int32') 
user_titlepos=np.array(user_titlepos,dtype='int32')
user_timeinterval=np.array(user_timeinterval,dtype='int32') 
user_age_label=np.array(user_age_label,dtype='float32') 
user_gender_label=np.array(user_gender_label,dtype='int32') 


import itertools
import random
results=[]
npratio=4
cutnum=2
keras.backend.clear_session()

MAX_SENT_LENGTH=30
MAX_SENTS=100


sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(word_dict), 300, trainable=True)
wordposition_emb=Embedding(MAX_SENT_LENGTH+1, 300, trainable=True)
embedded_sequences = embedding_layer(sentence_input)
wordpos_embedded_sequences = wordposition_emb(Lambda(lambda x:K.zeros_like(x,dtype='int32')+K.arange(x.shape[1]))(sentence_input))
embedded_sequences =add([embedded_sequences,wordpos_embedded_sequences ])
wordrep=Dropout(0.2)(embedded_sequences)


wordrep = Attention(16,16)([wordrep,wordrep,wordrep])
d_wordrep=Dropout(0.2)(wordrep)

attention = Dense(200,activation='tanh')(d_wordrep)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
l_att=Dot((1, 1))([d_wordrep, attention_weight])

textencoder = Model([sentence_input], l_att)


behavior_input = Input((MAX_SENTS,MAX_SENT_LENGTH,), dtype='int32') 
candidate_vecs = TimeDistributed(textencoder)(behavior_input)
behaviorpos_input = Input((MAX_SENTS,), dtype='int32') 
posembedding_layer = Embedding(MAX_SENTS+cutnum+1, 256, trainable=True)
posembedding=posembedding_layer(behaviorpos_input)

timeembedding_layer = Embedding(32, 256, trainable=True)

timeinterval= Input((MAX_SENTS,), dtype='int32') 
timeembedding=timeembedding_layer(timeinterval)
candidate_vecs=add([candidate_vecs,posembedding,timeembedding])
behaviorrep = Attention(16,16)([candidate_vecs,candidate_vecs,candidate_vecs])
d_behaviorrep=Dropout(0.2)(behaviorrep)
attentionu = Dense(200,activation='tanh')(d_behaviorrep)
attentionu = Flatten()(Dense(1)(attentionu))
attentionu_weight = Activation('softmax')(attentionu)
l_attu=Dot((1, 1))([d_behaviorrep, attentionu_weight])

model= Model([behavior_input,behaviorpos_input,  timeinterval], [l_attu])




mode='gender' #or 'age'
from sklearn.metrics import *
if mode == 'gender':
    model.load_weights('pretrainedum.h5')
    logits = Dense(2,activation='softmax')(Dense(200,activation='relu')(l_attu))
    modelfinetune= Model([behavior_input,behaviorpos_input,  timeinterval], [logits ])
    
    modelfinetune.compile(loss=['categorical_crossentropy'], optimizer=Adam(lr=0.0001), metrics=['acc'])
    

    for epoch in range(3):
        modelfinetune.fit([user_title[:16000],user_titlepos[:16000],user_timeinterval[:16000]] ,
                   [to_categorical(user_gender_label[:16000],2)],epochs=1,verbose=1,batch_size=64)
        pred = modelfinetune.predict([user_title[16000:18000],user_titlepos[16000:18000],user_timeinterval[16000:18000]],verbose=1)
        print(accuracy_score(np.argmax(pred,axis=-1) ,user_gender_label[16000:18000]))
        print(classification_report(np.argmax(pred,axis=-1) ,user_gender_label[16000:18000],digits=4))

    #static model
    model.load_weights('pretrainedum.h5')
    useremb=model.predict([user_title ,user_titlepos, user_timeinterval])
    useremb_input = Input((256,), dtype='float32') 
    logits2 = Dense(2,activation='softmax')(Dense(200,activation='relu')(useremb_input)
    modelstatic = Model([useremb_input], [logits2])
    modelstatic.compile(loss=['categorical_crossentropy' ], optimizer=Adam(lr=0.0001), metrics=['acc'])
    for ep in range(3):
        modelstatic.fit([useremb[:16000]] , [to_categorical(user_gender_label[:16000],2)],epochs=1,verbose=1,batch_size=64)
        pred = modelstatic.predict([useremb[16000:18000]],verbose=1)
        print(accuracy_score(np.argmax(pred,axis=-1) ,user_gender_label[16000:18000]))
        print(classification_report(np.argmax(pred,axis=-1) ,user_gender_label[16000:18000],digits=4))

if mode == 'age':
    model.load_weights('pretrainedum.h5')
    logits = Dense(4,activation='softmax')(Dense(200,activation='relu')(l_attu))
    modelfinetune= Model([behavior_input,behaviorpos_input,  timeinterval], [logits ])
    
    modelfinetune.compile(loss=['categorical_crossentropy'], optimizer=Adam(lr=0.0001), metrics=['acc'])
    

    for epoch in range(3):
        modelfinetune.fit([user_title[:16000],user_titlepos[:16000],user_timeinterval[:16000]] ,
                   [to_categorical(user_age_label[:16000],4)],epochs=1,verbose=1,batch_size=64)
        pred = modelfinetune.predict([user_title[16000:18000],user_titlepos[16000:18000],user_timeinterval[16000:18000]],verbose=1)
        print(accuracy_score(np.argmax(pred,axis=-1) ,user_age_label[16000:18000]))
        print(classification_report(np.argmax(pred,axis=-1) ,user_age_label[16000:18000],digits=4))
    
    
    #static model
    
    model.load_weights('pretrainedum.h5')
    useremb=model.predict([user_title ,user_titlepos, user_timeinterval])
    useremb_input = Input((256,), dtype='float32') 
    logits2 = Dense(4,activation='softmax')(Dense(200,activation='relu')(useremb_input)
    modelstatic = Model([useremb_input], [logits2])
    modelstatic.compile(loss=['categorical_crossentropy' ], optimizer=Adam(lr=0.0001), metrics=['acc'])
    for ep in range(3):
        modelstatic.fit([useremb[:16000]] , [to_categorical(user_age_label[:16000],4)],epochs=1,verbose=1,batch_size=64)
        pred = modelstatic.predict([useremb[16000:18000]],verbose=1)
    
        print(accuracy_score(np.argmax(pred,axis=-1) ,user_age_label[16000:18000]))
        
        print(classification_report(np.argmax(pred,axis=-1) ,user_age_label[16000:18000],digits=4))






#ctr task
with open('ads.tsv')as f:
    ads=f.readlines()


adsdict={}
for i in ads:
    tp=i.replace('\n','').split('\t')
    ttt=[word_dict[j][0] for j in word_tokenize(tp[1]) if j in word_dict][:30]
    ttd=[word_dict[j][0] for j in word_tokenize(tp[2]) if j in word_dict][:30]
    adsdict[tp[0]]=[ttt+[0]*(30-len(ttt)),ttd+[0]*(30-len(ttd))]  
    
with open('train.tsv')as f:
    adstrain=f.readlines()
with open('test.tsv')as f:
    adstest=f.readlines()   


user_title = []
user_titlepos = []
user_timeinterval = []
candidate_adstitle=[]
candidate_adsdes=[]
label=[]
MAXQ=100
MAXT=100
MAXQL=10
MAXTL=30
for i in adstrain:
    tm=i.replace('\n','').split('\t')
    tp=tm[5].split('#TAB#')
    newtp=[]
    lastindex=0
    tempcontent=''
    for i in range(len(tp)):

        if i>=1:
            if tp[i-1] in tp[i]:
                lastindex=i
                tp[i-1]=''
    tp=[x for x in tp if x!='']
    tp.reverse()
    for i in range(len(tp)):    

        if i>=1:
            if tp[i-1] in tp[i]:
                lastindex=i
                tp[i-1]=''
    tp.reverse()
    tr=[x for x in tp if x!=''] 
    title=[word_tokenize(x.lower()) for x in tm[7].split('#TAB#')]
    tt=[]
    for j in title:
        ttt=[]
        for m in j:
            if m in  word_dict :
                ttt.append( word_dict[m][0])
            else:
                ttt.append(1)
        ttt=ttt[:MAXTL]
        tt.append(ttt+[0]*(MAXTL-len(ttt)))
    tt=tt[:MAXT]
    tt=tt+[[0]*MAXTL]*(MAXT-len(tt))
    ttpos=list(range(1,len(tt)+1))+[0]*(MAXT-len(tt))
    lefttime=[0]+[int(np.log2(tm[8].split('#TAB#')[x+1][1]-tm[8].split('#TAB#')[x][1]+1)) for x in range(len(tm[8].split('#TAB#')[:100])-1)]
    righttime=lefttime[1:]+[0]  
    righttime+=[0]*(100-len(tt))
    
    pos=tm[2].split()
    neg=tm[3].split()
    for ad in pos:
        candidate_adstitle.append(adsdict[ad][0])
        candidate_adsdes.append(adsdict[ad][1])
        label.append(1)
        user_titlepos.append(ttpos)
        user_timeinterval.append(righttime)
        user_title.append(tt)
    for ad in neg:
        candidate_adstitle.append(adsdict[ad][0])
        candidate_adsdes.append(adsdict[ad][1])
        label.append(0)
        user_titlepos.append(ttpos)
        user_timeinterval.append(righttime)
        user_title.append(tt)


# In[54]:

 
user_title_test = []
user_titlepos_test = [] 
user_timeinterval_test = []
candidate_adstitle_test=[]
candidate_adsdes_test=[]
label_test=[]
for i in adstest:
    tm=i.replace('\n','').split('\t')
    tp=tm[5].split('#TAB#')
    newtp=[]
    lastindex=0
    tempcontent=''
    for i in range(len(tp)):

        if i>=1:
            if tp[i-1] in tp[i]:
                lastindex=i
                tp[i-1]=''
    tp=[x for x in tp if x!='']
    tp.reverse()
    for i in range(len(tp)):    

        if i>=1:
            if tp[i-1] in tp[i]:
                lastindex=i
                tp[i-1]=''
    tp.reverse()
    tr=[x for x in tp if x!=''] 
    title=[word_tokenize(x.lower()) for x in tm[7].split('#TAB#')]   
    tt=[]
    lefttime=[0]+[int(np.log2(tm[8].split('#TAB#')[x+1][1]-tm[8].split('#TAB#')[x][1]+1)) for x in range(len(tm[8].split('#TAB#')[:100])-1)]
    righttime=lefttime[1:]+[0]  
    righttime+=[0]*(100-len(tt))

    for j in title:
        ttt=[]
        for m in j:
            if m in  word_dict :
                ttt.append( word_dict[m][0])
            else:
                ttt.append(1)
        ttt=ttt[:MAXTL]
        tt.append(ttt+[0]*(MAXTL-len(ttt)))
    tt=tt[:MAXT]
    tt=tt+[[0]*MAXTL]*(MAXT-len(tt))
    ttpos=list(range(1,len(tt)+1))+[0]*(MAXT-len(tt))
    
    pos=tm[2].split()
    neg=tm[3].split()
    for ad in pos:
        candidate_adstitle_test.append(adsdict[ad][0])
        candidate_adsdes_test.append(adsdict[ad][1])
        label_test.append(1)
        user_timeinterval_test.append(righttime)
        user_titlepos_test.append(ttpos)
        user_title_test.append(tt)
    for ad in neg:
        candidate_adstitle_test.append(adsdict[ad][0])
        candidate_adsdes_test.append(adsdict[ad][1])
        label_test.append(0)
        user_timeinterval_test.append(righttime)
        user_titlepos_test.append(ttpos)
        user_title_test.append(tt)


# In[55]:


 
user_title=np.array(user_title,dtype='int32') 
user_titlepos=np.array(user_titlepos,dtype='int32')  
user_timeinterval=np.array(user_timeinterval,dtype='int32')  
candidate_adstitle=np.array(candidate_adstitle,dtype='int32') 
candidate_adsdes=np.array(candidate_adsdes,dtype='int32') 
label=np.array(label,dtype='float32') 
 
user_title_test=np.array(user_title_test,dtype='int32') 
user_titlepos_test=np.array(user_titlepos_test,dtype='int32') 
user_timeinterval_test=np.array(user_timeinterval_test,dtype='int32') 
candidate_adstitle_test=np.array(candidate_adstitle_test,dtype='int32') 
candidate_adsdes_test=np.array(candidate_adsdes_test,dtype='int32') 
label_test=np.array(label_test,dtype='float32') 








model.load_weights('0pretrainmulti0.h5')

trainvalindex=list(np.where(label==1)[0])+random.sample(list(np.where(label==0)[0]),len(np.where(label==1)[0]))
random.shuffle(trainvalindex)
trainindex=np.array(trainvalindex[:int(len(trainvalindex)*0.9)])
valindex=np.array(trainvalindex[int(len(trainvalindex)*0.9):])


ad_input = Input(shape=(MAX_NEWS_LENGTH,), dtype='int32')
ad_input2 = Input(shape=(MAX_NEWS_LENGTH,), dtype='int32')

embedded_sequences2 = embedding_layer(ad_input)
wordrep2=Dropout(0.2)(embedded_sequences2)
embedded_sequences3 = embedding_layer(ad_input2)
wordrep3=Dropout(0.2)(embedded_sequences3)
 

wordrep2 = Attention(16,16)([wordrep2,wordrep2,wordrep2])
d_wordrep2=Dropout(0.2)(wordrep2)

attentionw2 = Dense(200,activation='tanh')(d_wordrep2)
attentionw2 = Flatten()(Dense(1)(attentionw2))
attention_weightw2 = Activation('softmax')(attentionw2)
l_atttitle=Dot((1, 1))([d_wordrep2, attention_weightw2])

    
wordrep3 = Attention(16,16)([wordrep3,wordrep3,wordrep3])
d_wordrep3=Dropout(0.2)(wordrep3)

attentionw3 = Dense(200,activation='tanh')(d_wordrep3)
attentionw3 = Flatten()(Dense(1)(attentionw3))
attention_weightw3 = Activation('softmax')(attentionw3)
l_attdes=Dot((1, 1))([d_wordrep3, attention_weightw3])
advecs =concatenate([Lambda(lambda x: K.expand_dims(x,axis=1))(vec) for vec in [l_atttitle,l_attdes]],axis=1)
attentionadv= Dense(200,activation='tanh')(advecs)
attentionadv = Flatten()(Dense(1)(attentionadv))
attention_weightadv = Activation('softmax')(attentionadv)
unifiedad=Dot((1, 1))([advecs, attention_weightadv])
score = Activation(keras.activations.sigmoid)(dot([l_attu, unifiedad], axes=-1))
modelfinetune = Model([ad_input,ad_input2,behavior_input,behaviorpos_input,timeinterval], score)

    
modelfinetune.compile(loss=['binary_crossentropy'], optimizer=Adam(lr=0.0001), metrics=['acc'])
    
for ep in range(3):
    
    modelfinetune.fit([candidate_adstitle[trainindex],candidate_adsdes[trainindex],user_title[trainindex],user_titlepos[trainindex],user_timeinterval[trainindex]],label[trainindex],shuffle=True,batch_size=64, epochs=1,verbose=1)
    pred = modelfinetune.predict([candidate_adstitle[valindex],candidate_adsdes[valindex],user_title[valindex],user_titlepos[valindex],user_timeinterval[valindex]],verbose=1)
    #pred = modelfinetune.predict([candidate_adstitle_test,candidate_adsdes_test,user_title_test,user_titlepos_test,user_timeinterval_test],verbose=1)

    all_auc=roc_auc_score(label_test,pred)
    all_ap=average_precision_score(label_test,pred)
    print(all_auc,all_ap)


#static model
model.load_weights('pretrainedum.h5')
useremb=model.predict([user_title ,user_titlepos, user_timeinterval])
useremb_test=model.predict([user_title_test ,user_titlepos_test, user_timeinterval_test])
useremb_input = Input((256,), dtype='float32') 
score = Activation(keras.activations.sigmoid)(dot([useremb_input, unifiedad], axes=-1))
modelstatic = Model([ad_input,ad_input2,useremb_input], [logits2])
modelstatic.compile(loss=['categorical_crossentropy' ], optimizer=Adam(lr=0.0001), metrics=['acc'])
for ep in range(3):
    
    modelstatic.fit([candidate_adstitle[trainindex],candidate_adsdes[trainindex],useremb[trainindex]],label[trainindex],shuffle=True,batch_size=64, epochs=1,verbose=1)
    pred = modelstatic.predict([candidate_adstitle[valindex],candidate_adsdes[valindex],useremb[valindex]],verbose=1)
    #pred = modelstatic.predict([candidate_adstitle_test,candidate_adsdes_test,useremb_test],verbose=1)

    all_auc=roc_auc_score(label_test,pred)
    all_ap=average_precision_score(label_test,pred)
    print(all_auc,all_ap)

