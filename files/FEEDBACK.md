### Feedback on the Project Proposal

This proposal is OK, it's a Go. Just some comments:

Generating song lyrics based on the datasets is definitely out of scope, that would be a whole different project.

The sophisticated features you mention are a bit unclear (what is "orientation" for example) and so is how you would be extracting them from the data. I understand this is from the paper by Fell and Sporleder, but more explanation would have been good.

For the dataset, consider leaving out the less frequent genres to make sure the dataset is not reduced too much after balancing.

GTZAN dataset: in your final report, please include the link as a footnote.

Evaluation metrics: it's definitely a good idea to use F1 either way, but here you say it's because "some variation of our data will have high imbalance" but in the previous section you said you were going to balance your dataset. So that sounds contradictory. Maybe it's because you don't clarify what you mean by "some variation". 

Finally, note that your proposal title says Group 42, but according to Canvas you are actually Project Group 43.