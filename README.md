# MyLLM

Self implemented GPT2, with the help of original paper "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) and "Language Models are Unsupervised Multitask Learners" (https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

Also, for specific formulas and architecture details I looked at Andrew's Karpathy NanoGPT (https://github.com/karpathy/nanoGPT/blob/master/model.py) As well as original OpenAI's git (https://github.com/openai/gpt-2).

I like implementing everything from scratch, so this project implements BPE tokenizer (from GPT2 paper) as well as Transformer architecture composed from simple Pytorch structures.

I will also try to implement some nice analysis to take inside into gpt work, attention values etc.