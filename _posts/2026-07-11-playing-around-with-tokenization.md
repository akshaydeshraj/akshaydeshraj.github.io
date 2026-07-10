---
layout: post
title: "Playing around with tokenization"
date: 2026-07-11 18:00:00 +0530
tags: [tokenization, llms]
---

I spent some time this week on a small tokenizer puzzle: build one tokenizer for the India page on Wikipedia across English, Hindi, Telugu, and Malayalam.

The metric was fertility:

`fertility = tokenizer tokens / words`

Lower is better, but the real trap is unevenness. The final score depends on the gap between the best and worst language. English also has to stay below 1.2.

I used BPE with byte fallback. That part is non-negotiable for me because there should be no unknown token. If the tokenizer sees a character it has not learned as a normal token, it can still encode the UTF-8 bytes.

After training the BPE, I used the remaining vocabulary budget for frequent word-shaped tokens from the same pages. That is not a general tokenizer design. It is very much shaped around the evaluation text, which is the point of the exercise.

Current run:

| Language | Words | Tokens | Fertility |
| --- | ---: | ---: | ---: |
| English | 10,363 | 12,433 | 1.1997 |
| Hindi | 15,709 | 18,357 | 1.1686 |
| Telugu | 7,370 | 8,509 | 1.1545 |
| Malayalam | 11,191 | 13,090 | 1.1697 |

Vocab size: 10,000  
Unknown tokens: 0  
Fertility spread: 0.0452  
Score: 22,122.11

Artifacts:

- [tokenizer.json](/assets/tokenizers/tokenizer.json)
- [stats.json](/assets/tokenizers/stats.json)

The useful bit for me was seeing the split clearly: byte fallback buys coverage, BPE buys compression, and the last few thousand tokens buy balance against the specific text you expect to see.
