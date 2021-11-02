# NonsenseDetector
Attempting to train an algorithm which can reliably distinguish whether an input makes sense

To see the algorithm in its current form, enter into the command shell:

> python nonsense_bot.py

This will prompt the user for a sentence input. Once the user types an input sentence and hits enter, the sentence will be sent through three separate models: a fully-connected network (labelled "baseline"), a convolutional neural network ("cnn"), and a recurrent neural network ("rnn"). The convolutional neural network may sometimes be skipped, as it currently triggers errors if used on very short sentences.

Each of the three models will output a label, "sense" or "nonsense". The console will print out the labels predicted by all three models, as well as the probability of the correct label being "nonsense". To be clear, even if the model predicts that the correct label is "sense", the following decimal still represents the probability of the correct label being "nonsense". An example output might look like:

> Enter a sentence:
> What once seemed creepy now just seems campy
> Model baseline:  sense  ( 0.419 )
> Model cnn:  sense  ( 0.363 )
> Model rnn:  nonsense  ( 0.992 )
