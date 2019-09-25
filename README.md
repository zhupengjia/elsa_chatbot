## This repo is for ElsaBot. 

ElsaBot is a general purpose chatbot as a human-machine dialogue interface for other services. It allows people to interact with each backend service via natural language. Elsa chatbot not only allows to do several different types of jobs at a time, but also allows to reply a human-like emotional response. An easy-to-use dialog-design system can also be used for designing to do a custom job. Elsa chatbot can be embedded into several different business scenarios, such like customer service, recommondation, investment assistant, risk alert, system monitor.

## Model
The general framework is: 
![elsabot](https://github.com/zhupengjia/elsa_chatbot/blob/develop/docs/img/chatbot.png?raw=true)

### Dialog Status
When a user query come, the query will be tokenized and spell checked, the related entities will also be extracted. A sentiment analyzer is used to get the sentiment of query. After all, the intermediate data, tokens, sentiment, entities, will be saved to a global register: dialog status. The dialog status saves any variables generated in a whole dialog session, besides the variables mentioned above, it also saves variables got from function call. The dialog status will then be used as input for further chatbot engines.

### Topic Manager
Elsabot aims to process different domains at the same time, and it  mixes several chatbot engines for different purpose. Topic manager is to identify topics via conversion and identify human behavior of changing topics. Here topic manager is able to understand users’ inquires and allocate them to different domains. Specifically, the topic manager in our solution can either be a text classifier for deciding whether switch topics, or as simple as sequential decide which engine will be used based on the score returned from each engine.

### Chatbot engines
Each chatbot engine uses the same input (dialog status) and output that can easily be mixed via topic manager. Currently ElsaBot includes several engines: rule retrieval based, end2end goal oriented chatbot engine, controllable generative based chatbot engine, knowledge based chatbot, read comprehension QA from raw texts.

#### Rule-retrieval based chatbot engine
The rule-retrieval based chatbot engine reads predefined rules from an excel speadsheet. An example of rule file can be found in [[link]](https://github.com/zhupengjia/elsa_chatbot/blob/develop/config/example_rule.xlsx). With a simple designed rule interface, user can easily define their dialog flows. In the rule definition, each intent has multiple user_says, which indicates to some possible query with same meaning; and multiple response for randomly return. The entity mask provides additional control for intent search range, and with defined child id, the intent search scope will be limited for next loop dialog. In each turn the chatbot engine will do the semantic search for the most closed user says in the search scope, and then call the defined function. The predefined response in each intent support to use mask in order to fill variables saved in dialog status.

The benefit of the rule-retrieval based chatbot is its simple design, and can easily be used for a chatbot warm start. For the better performance for dialogue policy, an end2end chatbot engine is used. 

#### End2end goal oriented chatbot engine
The architecture of end2end goal oriented chatbot engine is:
![end2end](https://github.com/zhupengjia/elsa_chatbot/blob/develop/docs/img/end2end_chatbot.png?raw=true)

It uses the utterance and entity information from dialog status. The utterance goes through a transformer based sentence encoder, while the entity names goes through a one hot embedding and several dense layers. For better performence, pretrained BERT model can be used for fine-tuning. The information of them are combined and pass through a LSTM network with the softmax activation function, and classify to a predefined response. Use the entity names for response mask, use can manually predefine the response policy, for example, some response need some entities exist before chosen. 

The end2end chatbot engine share the same config file(rule file) with rule-retrieval based chatbot engine, it can be used while user accumulated enough chat logs.

Beside the supervised learning from chat logs, the end2end chatbot engine support to use policy gradient based reinforcement learning. The policy gradient can either be trained via online training, or with a user simulator. Currently the user simulator can be designed via a rule-based chatbot engine.

#### Controllable generative based chatbot engine
The generative based chatbot engine is based on sequence to sequence framework:
![generative](https://github.com/zhupengjia/elsa_chatbot/blob/develop/docs/img/generative_chatbot.png?raw=true)

In general, the generative based chatbot engine contains transformer based encoder and decoder. For doing the controllable response generation, the predefined sentiment and the entitie names are embedded and concatenated to the hidden layers. 

Together with the goal oriented chatbot engine, the generative based chatbot engine makes dialogues with more fun.  An emotional response can let people feels more humanity, thus can provide better user experience.

A simple demo:
![gendemo](https://github.com/zhupengjia/elsa_chatbot/blob/develop/docs/img/demo_generative.png?raw=true)

#### Knowledge based chatbot engine
Knowledge Graph (KG) is widely considered as an effective way of expanding inquiries. Specifically, the graph consists of a collection of head-relation-tail triples which contains two related topics, head, tail and  its sequence for example (Einstein, Relativity) and (Newton, Gravity). With the assist of KG, Chatbot is able to response to inquiries in a comprehensive way especially when keywords are not listed in inquiries.

ElsaBot uses a simple method for KBQA:  First the node and relation  are extracted via keyword based entity recognition, then multi relations extracted via part of speech. ElsaBot use the existed Spacy package to finish the above tasks. Then the knowledge graph embedding was trained via RotatE model. The final answer was extracted from the knowledge graph embedding. 

Using the knowledge graph embedding for graph query has the ability to  find the hidden relationships in the graph, also easier to implement. However, there are two serious weaknesses using the knowledge graph embedding via RotatE:  bad performance for multi-relation, no ability to cover new nodes and relations. A reliable solution is using a traditional SPARQL query of knowledge graph. 

#### Read comprehension engine
There exists large mount of unstructured texts in the world or in the company. Label them to structured data such like knowledge graph or question-answer pairs sometimes is a huge and expensive task. Currently several high quality read comprehension datasets are released,  and people hope one day the machine can really understand the documents and question, and reply the reasonable high level answers. Some challenges shows that the machine can retrieve better performance than human. However, due to the limitation of the datasets, there is still a very long way to make machine really understand our languages.

The read comprehension engine for ElsaBot tries to get answer from raw texts directly. The framework:
![comprehension](https://github.com/zhupengjia/elsa_chatbot/blob/develop/docs/img/ReadComprehension.png?raw=true)
First the raw documents are indexed by Xapian, people can also choose other search engines to replace Xapian. Once a query comes, the search engine first get the related texts from raw documents, then the model extract the answer from the prefiltered texts. The model is fine-tuned from XLNet, and trained using SQuAD2 datasets. 

A simple demo:
![rcdemo](https://github.com/zhupengjia/elsa_chatbot/blob/develop/docs/img/demo_rc.png?raw=true)

#### Resful api based chatbot engine
ElsaBot has a flexible framework that support to use other chatbot engines via restful api. User can easily extend the ability of chatbot and get better user experience. Also it is easier to extend the load capacity of chatbot with the usage of container service.

## IT Architecture
TODO


