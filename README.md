This repo is for end2end version chatbot, For more details please check [README](http://n2.c3.acnailab.com/code/chatbot-end2end/index.html)

The chatbot uses a LSTM for dialog tracker to support multi-turn dialog. From now we implemented a supervised learning, with input of dialog embedding, and output of response id. Futher will integrate reinforcement learning. The dialog embedding are from current utterance, previous presonse and current entities. And the response id from predefined response template. The general framework is: 
![end-2-end](http://n2.c3.acnailab.com/code/chatbot-end2end/pic/end2end.png)

The code reserved several readers for different sources, such like api.ai like json file, etc. Also the code reserved a sentence embedding connector for further transfer learnings of sentence embedding. The chatbot uses entity masks in response template to support manually control the dialog flow, some tests in previous show that this can significantly improve the performance. The code also integrated the hook functions to the neural network via dialog status.

The entity recognition, is provided in ailab package. Please check ailab package for more details.

Please read docs/end2end.mindnode for the general framework. You need to install mindnode5 to read this file.

Instead, you can also read documents/end2end.png for general framework, and the notes for each node can be found in docs/end2end.md

Related paper:

* http://arxiv.org/abs/1606.01269
* http://arxiv.org/abs/1702.03274
* http://arxiv.org/abs/1711.01731
