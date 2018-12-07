# Friendroid- A RNN based chatbot

Friendroid is a chatbot using RNN to predict responses for human interaction. This is based on Sequence-to-sequence model. Salient features: LSTM cells, bidirectional dynamic RNN, decoders with attention.

Project objectives are as follows:

• Personalizing the voice for the bot.
• Speech to text and text to speech functionality.
• Chatter bot learns from the database.
• GUI for the bot

Database used: Cornell MovieDialogs Corpus

Approach

• We used the data set named ”Cornell MovieDialogues
Corpus”.
• Trained the model using RNN with LSTM.
• Integrated GUI using Tkinter python.
• Incorporated features like speech to text and text to
speech in python

ANALYSIS

• Accuracy of our model on test data is around 43 %.
• The learning curve shows that training loss decreases as
the number of epochs increase. It shows that the model
is continuously learning.
– The model learns faster initially and reaches a constant state eventually.
• Our model gives more meaningful responses on longer
sentences than an shorter ones. Since our model does
not remember context of the conservation, it may give
illogical answers on certain questions of small length.
• According to 20 human evaluators, our model performs
moderately in giving sensible responses. The model gives
incorrect and illogical responses at times, when the question contains words not present in the model vocabulary.
• F1 score is 0.54. Precision value for our model is higher
than recall.Precision is high when words of request and
response and present in the model vocabulary.
• Our model has good results when all words used in request and response are present in the model vocabulary.If
not, our model is prone to give illogical results.
