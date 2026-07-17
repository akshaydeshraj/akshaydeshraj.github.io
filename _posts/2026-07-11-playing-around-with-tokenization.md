---
layout: post
title: "Playing around with tokenization"
date: 2026-07-11 00:01:00 +0530
tags: [tokenization, llms]
hidden: true
sitemap: false
---

I built a tokenizer because "tokens" is one of those words that feels obvious until it touches Hindi, Telugu, URLs, brackets, and a Wikipedia table.

The setup: take India's Wikipedia page in English, Hindi, Telugu, and Maithili. Convert the HTML to Markdown without quietly dropping visible text. Train one shared BPE tokenizer with a 10,000-token vocabulary.

The tokenizer is deliberately boring: Hugging Face BPE, no text normalizer, Metaspace pre-tokenization. The exported `tokenizer.json` has 10,000 entries in the actual BPE vocab. I did not use a custom encoder or added-token padding.

The score I cared about was balance. Each language gets this ratio:

`ratio = tokenizer tokens / faithful units`

Faithful units are the visible pieces of the text. A word counts as one unit. A punctuation mark or symbol counts as one unit. So a Markdown link is counted as the link text plus the brackets, slashes, dots, and URL chunks around it. Slightly annoying, but fair. The tokenizer has to encode those characters too.

<div class="tokenizer-tables">
  <table class="data-table numeric-table">
    <caption>Current run</caption>
    <thead>
      <tr>
        <th scope="col">Language</th>
        <th scope="col" class="num">Faithful units</th>
        <th scope="col" class="num">Tokens</th>
        <th scope="col" class="num">Ratio</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th scope="row">English</th>
        <td class="num">186,367</td>
        <td class="num">112,223</td>
        <td class="num">0.6022</td>
      </tr>
      <tr>
        <th scope="row">Hindi</th>
        <td class="num">88,359</td>
        <td class="num">53,990</td>
        <td class="num">0.6110</td>
      </tr>
      <tr>
        <th scope="row">Telugu</th>
        <td class="num">36,292</td>
        <td class="num">21,561</td>
        <td class="num">0.5941</td>
      </tr>
      <tr>
        <th scope="row">Maithili</th>
        <td class="num">5,808</td>
        <td class="num">3,473</td>
        <td class="num">0.5980</td>
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
        <td class="num">0 on corpus</td>
      </tr>
      <tr>
        <th scope="row">Spread</th>
        <td class="num">0.0169</td>
      </tr>
      <tr>
        <th scope="row">Score</th>
        <td class="num">59,059.32</td>
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
