This repo is for elsa chatbot, For more details please check [this presentation](https://nlp-lab.com/chatbot)

Elsa chatbot is a general purpose chatbot as a human-machine dialogue interface for other services. It allows people to interact with each backend service via natural language. Elsa chatbot not only allows to do several different types of jobs at a time, but also allows to reply a human-like emotional response. An easy-to-use dialog-design system can also be used for designing to do a custom job. Elsa chatbot can be embedded into several different business scenarios, such like customer service, recommondation, investment assistant, risk alert, system monitor.

The general framework is: 
![elsabot](https://github.com/zhupengjia/elsa_chatbot/blob/develop/docs/img/chatbot.png?raw=true)

The code reserved several readers for different sources, such like api.ai like json file, etc. Also the code reserved a sentence embedding connector for further transfer learnings of sentence embedding. The chatbot uses entity masks in response template to support manually control the dialog flow, some tests in previous show that this can significantly improve the performance. The code also integrated the hook functions to the neural network via dialog status.

Related paper:

* http://arxiv.org/abs/1606.01269
* http://arxiv.org/abs/1702.03274
* http://arxiv.org/abs/1711.01731
