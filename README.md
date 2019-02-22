This repo is for end2end version chatbot, For more details please check [README](http://wiki.higgslab.com/chatbot-end2end/index.html)

Elsa chatbot is a general purpose chatbot as a human-machine dialogue interface for other services. It allows people to interact with each backend service via natural language. Elsa chatbot not only allows to do several different types of jobs at a time, but also allows to reply a human-like emotional response. An easy-to-use dialog-design system can also be used for designing to do a custom job. Elsa chatbot can be embedded into several different business scenarios, such like customer service, recommondation, investment assistant, risk alert, system monitor.

Challenge:
1. Traditional retrieval based dialog systems are composed of several modules, usually each is separately hand-crafted or trained. They may require to predefine the structure of the dialog status as a form composed of a set of slots to be filled during the dialog, and are inherently limited by their design. Elsa chatbot gives an end-to-end solution based on neural network with minimal hand-crafting of state, and no design of a dialog act taxonomy is necessary. 
2. An important feature of chatbot is evolvability. Elsa chatbot uses a supervised learning system to allow learning from history dialogs, and a reinforcement learning system to adaptively learning after deployment to improve the performance based on weak signals.
3. Elsa chatbot allows to train several different skills seperately and integrate them together. This allows chatbot can deal with different jobs at the same time, and also allow to reply a human-like response if there is no job correlated instead of replying a general response.  

Currently the chatbot includes a goal-oriented dialog engine and a generative based dialog engine, managed by a topic manager. A global dialog status allows to track the current status or variables such like entities, sentiment, current topic, history dialogs. The goal-oriented dialog engine uses a LSTM dialog tracker to support multi-turn dialog. The generative based engine uses a transformer based seq2seq model with additional dialog status input in hidden layer for controllable response generation. Two dialog engines share the same sentence encoder, use the pretrained BERT model.

The general framework is: 
![end-2-end](http://wiki.higgslab.com/chatbot-end2end/pic/end2end.png)

The code reserved several readers for different sources, such like api.ai like json file, etc. Also the code reserved a sentence embedding connector for further transfer learnings of sentence embedding. The chatbot uses entity masks in response template to support manually control the dialog flow, some tests in previous show that this can significantly improve the performance. The code also integrated the hook functions to the neural network via dialog status.

Detailed for each modules in framework please check: [end2end modules](http://code.higgslab.com/nlp/chatbot-end2end/src/branch/master/docs/end2end.md)


Related paper:

* http://arxiv.org/abs/1606.01269
* http://arxiv.org/abs/1702.03274
* http://arxiv.org/abs/1711.01731
