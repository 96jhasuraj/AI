# Experiment 1 : 76%
- did not implement masking for tokens 
    - this was a bug , in attention all the padding would be treated as a token 
- achieved 76% test accuracy

# Experiment 2 : 77%
- implemented masking for tokens
- still 77% test accuracy

# Experiment 3 : **82.5%**
- So i was using mean averaging in the end , but while reading stuff i realized bert encoder uses a token at start whose role is just to get influnced by all other tokens [CLS]
    - this was a better strategy than mean pooling since good bad will get averaged out , getting influenced by other tokens & patterns is definetly a better strategy
- I also increased the max token length :
    - My GPU wont support very big context length without other optimizations 
    - Yet to read details of optims like flashattention & kvcache . In future experiments i will try these + quantization of model as well
- Also reduced the learning rate by a factor of 10 + changed scheduler 
- Accuracy : 82.5% (a 5.5% JUMP , i think majority of it comes from changing the pooling strategy)

