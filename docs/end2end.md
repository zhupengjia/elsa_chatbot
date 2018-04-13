# Entity  extraction

Provided from ailab package:  
ailab.text.ner.py


## Keywords

Related config:  
ner:  
	keywords:  
		entity_name1: keyword_list1  
		...  
	annoy_filter: filter threshold (for vector search)  
  
Related code:  
ailab/text/ner.py  
Class NER_Base:  
	get_keywords  
  
The keywords search will also migrate to regex search in the code  

## Regex

Related config:  
ner:  
	regex:  
		entity_name1: regex1  
		....  
  
Related code:  
ailab/text/ner.py  
Class NER_Base:  
	get_regex

## NER

Related config:  
ner:  
	ner:  
			entity_name1:entity_raw_name_from_extraction  
		....  
  
Related code:  
ailab/text/ner.py  
Class NER_Base:  
	get_ner


# Entity dictionary

Location:  
src/module/entity_dict.py  
  
Purpose: entity dictionary, convert entity names and values to ids  
  
The __call__ function will convert entity from entityname:[entityvalues] to entityname_id:[entityvalue_ids]  
  
Two dictionaries are in this dict:  
entity name to id  
entity value to id  
  
There are also maintained a entity_mask dict, which is used to convert existed entities to a one-hot array


# Tokenizer

Provided from ailab package:  
ailab.text.ner  
ailab.text.tokenizer


# Response dictionary

Location:  
src/module/response_dict.py  
  
Purpose: response dictionary  
Combined with the function 'build_responses' and 'get_response' in src/reader/reader_base.py   
  
Related config:  
response_template:   
    data: response template loc  
    cached_index: indexfile  
    cached_vocab: vocabfile  
    vocab_size: response_vocab_size  
    ngrams: tfidf ngram  
  
First you need a response template file, the format in each line is:  
needed_entity | notneeded_entity | func_call | response  
needed_entity means this response is available only those entities existed  
notneeded_entity means this response is not available if those entities existed  
func_call is the needed function call  before return the response. The available func_call is in src/hook/behaviors. In the future will support web hooks  
  
The class will build a tf-idf index for template, the __getitem__ method is to get the most closed response via the tf-idf algorithm.(only used for training, the response string in training data will convert to a response id via tfidf search)  
  
The function 'build_mask' in the class is used to build the entity mask, converted from the template


# Dialog Pairs

List of dialogs:  
[  
	[  
		[utterance, response],  
		...  
	],  
	...  
]


## Reader_VUI

read from VUI json files (compatible with old api.ai format)  
Location:  
src/reader/reader_vui.py

## Reader_Babi

read from babi training data  
Location:  
src/reader/reader_babi.py

## Reader_Dialog

read from dialog history (HR chatbot history)  
Location:  
src/reader/reader_dialog.py

## Reader_YAML

Read from yaml dialogs  
Location:  
src/reader/reader_yaml.py


# Dialog predeal

Pre-dealt dialog:  
{  
	'utterance': [utterance token_ids],  
	'response': [response ids]  
	'ent_utterance':[utterance entity ids]  
	'idrange':[  
		[dialog_startid, dialog_endid],  
		...  
	]  
}  
  
Location:  
src/reader/reader_base.py


## Utterance entity_ids

## Response ids

## Utterance token_ids


# Vocab Dictionary

Provided from ailab package:  
ailab.text.vocab  
  
tokens to id 


# Response Encoder

Convert response one-hot array to embedding, mlp


# Predicted response


## softmax


# Dialog Tracker

Location:  
src/model/dialog_tracker.py  
  
Use lstm for the dialog tracker


# Dialog_Embedding


# Dialog Status

Location:  
src/module/dialog_status.py  
  
Maintain the dialog status in a dialog  
  
The dialog status will maintain:  
utterance history  
response history  
entity mask history  
entity history  
current entity mask  
current entities  
  
The class also provide the method to convert those data to torch variables  
  
To use this class, one must do the following steps:  
add_utterance: add the current utterance to status, the entities will be extracted from utterance  
getmask: create entity mask from existed entities  
add_response: add the current response to status, and run the corresponded hook functions


## previous response

## mask

## entities(names)

## batch_size

To speed up, the lstm tracker use pack_padded_sequence to pack several dialogs together

## response

## utterance


# Entity Encoder

Convert entity names one-hot array to embedding, mlp


# Supervised Learning Loss function

Please check the code:  
  
src/model/supervised.py


# Sentence Encoder

Sentence embedding  
  
Source:  
src/model/sentence_encoder.py  
  
Use CNN as sentence embedding  
  
People can replace this encoder to another one. For example the pretrained sentence embedding, the encoder part from other models, etc


## pretrained encoder

- duplicate QA embedding
  Location:  
  src/model.duplicate_embedding.py  
    
  Use Quora duplicate QA dataset for now

- autoencoder
  Location:  
  src/model.self_embedding.py  
    
  predict sentence itself

- language model


# Hooks

input is the current entities  
output is variables, will merge to current entities


## webhooks

not implemented

## Embedded functions

Location:  
src/hook/behaviors.py


# Rule Based

Rule based chatbot  
  
code:  
src/model/rule_based.py


## Reader_VUI

Read rules from webUI  
Not implemented

## Reader_XLSX

Read rules from excel table  
Location:  
src/reader/rulebased/reader_xlsx.py  
  
excel Format:  
first row:  
id : the intent id  
childID : the child intent id for this intent, if empty will search from all intents  
userSays : all possible usersays, splitted by \n. If empty will dealt as the first greeting say.  
response : response sentence  
webhook : callable hook function. if 'RESPONSE' key in return will reply correlated value


# User Simulator

An example of babi simulator please check the hook function:  
src/hook/babi_gensays.py  
  
and the rules:  
data/babi/babi_adversary.xlsx


# Policy Gradiant Reinforcement Learning

Please check the code:  
  
src/model/policy_gradiant.py


# Real User

