## Been reading about SFT , DPO , GRPO 

Observations
1. So GRPO without SFT is a bad idea. I ended up restarting my notebook & for a while was traning using GRPO without SFt. Found out the issue because model's reward for spelling was always towards max negative . Looking at the outputs , random extra characters were added after the expected word.


Learning:

1. Having a debug reward function is good idea till I decide on a specific configuration to train model


TO READ:
1. how do we check for overfitting in GRPO / DPO ? need to read a bit