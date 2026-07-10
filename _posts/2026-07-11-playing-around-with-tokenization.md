---
layout: post
title: "Playing around with tokenization"
date: 2026-07-11 00:01:00 +0530
tags: [tokenization, llms]
---

I spent some time this week on a small tokenizer puzzle: build one tokenizer for the India page on Wikipedia across English, Hindi, Telugu, and Malayalam.

The metric was fertility:

`fertility = tokenizer tokens / words`

Lower is better, but the real trap is unevenness. The final score depends on the gap between the best and worst language. English also has to stay below 1.2.

I used BPE with byte fallback. That part is non-negotiable for me because there should be no unknown token. If the tokenizer sees a character it has not learned as a normal token, it can still encode the UTF-8 bytes.

After training the BPE, I used the remaining vocabulary budget for frequent word-shaped tokens from the same pages. That is not a general tokenizer design. It is very much shaped around the evaluation text, which is the point of the exercise.

<div class="tokenizer-tables">
  <table class="data-table numeric-table">
    <caption>Current run</caption>
    <thead>
      <tr>
        <th scope="col">Language</th>
        <th scope="col" class="num">Words</th>
        <th scope="col" class="num">Tokens</th>
        <th scope="col" class="num">Fertility</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">English</th>
        <td class="num">10,363</td>
        <td class="num">12,433</td>
        <td class="num">1.1997</td>
      </tr>
      <tr>
        <th scope="row">Hindi</th>
        <td class="num">15,709</td>
        <td class="num">18,357</td>
        <td class="num">1.1686</td>
      </tr>
      <tr>
        <th scope="row">Telugu</th>
        <td class="num">7,370</td>
        <td class="num">8,509</td>
        <td class="num">1.1545</td>
      </tr>
      <tr>
        <th scope="row">Malayalam</th>
        <td class="num">11,191</td>
        <td class="num">13,090</td>
        <td class="num">1.1697</td>
      </tr>
    </tbody>
  </table>

  <table class="data-table summary-table">
    <caption>Summary</caption>
    <tbody>
      <tr>
        <th scope="row">Vocab size</th>
        <td class="num">10,000</td>
      </tr>
      <tr>
        <th scope="row">Unknown tokens</th>
        <td class="num">0</td>
      </tr>
      <tr>
        <th scope="row">Fertility spread</th>
        <td class="num">0.0452</td>
      </tr>
      <tr>
        <th scope="row">Score</th>
        <td class="num">22,122.11</td>
      </tr>
    </tbody>
  </table>

  <table class="data-table artifact-table">
    <caption>Artifacts</caption>
    <tbody>
      <tr>
        <th scope="row">Tokenizer</th>
        <td><a href="/assets/tokenizers/tokenizer.json" download="tokenizer.json">tokenizer.json</a></td>
      </tr>
      <tr>
        <th scope="row">Run stats</th>
        <td><a href="/assets/tokenizers/stats.json" download="stats.json">stats.json</a></td>
      </tr>
    </tbody>
  </table>
</div>

The useful bit for me was seeing the split clearly: byte fallback buys coverage, BPE buys compression, and the last few thousand tokens buy balance against the specific text you expect to see.
