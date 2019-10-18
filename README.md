# HDL-CRF

We update our dataset in recently,this is BNER-2.0   2019-6

This is the code of the paper 《A Hybrid Deep Learning Framework for Bacterial Named Entity Recognition》

This is  a model for bacterial name entity recognition

you should get a pre-trained  word embedding  in data/ .

such as word2vec:http://bio.nlplab.org/#word-vector-tools


you can train your own model through the train.py or use pre-trained model to predict 

the parameter file in model/config.py,you can adjust the parameter .
the model weights of ours in test file

Firstly, you should create a directory data and download or train a word embedding ; then put the word embedding in data direcctory and  revise the path of word embedding file in config.py .

1.Train your own model
  1) python build_data.py
  2) python train.py
  3) python evaluate.py
 
2. use pre-trained model
if you do not have the word2vec.6b.300d.npz, you should run python build_data.py to generate this file.
 1) python predict.py
 
3. evaluate the model of us

    python evaluate.py
    
    
  
references:

https://github.com/guillaumegenthial/sequence_tagging

