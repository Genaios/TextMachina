🛠️ Supported tasks
========

![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true) can generate datasets for MGT detection, attribution, boundary detection, and mixcase detection:

<p align="center">
  <picture>
    <img alt="CLI interface showing generated and human text for detection" src="https://github.com/Genaios/TextMachina/blob/main/assets/tasks/detection.png?raw=true">
    <figcaption>Example from a detection task.</figcaption>
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
  <picture>
    <img alt="CLI interface showing generated and human text for attribution" src="https://github.com/Genaios/TextMachina/blob/main/assets/tasks/attribution.png?raw=true">
    <figcaption>Example from an attribution task.</figcaption>
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
  <picture>
    <img alt="CLI interface showing generated and human text for boundary" src="https://github.com/Genaios/TextMachina/blob/main/assets/tasks/boundary.png?raw=true">
    <figcaption>Example from a boundary detection task.</figcaption>
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
  <picture>
    <img alt="CLI interface showing generated and human text for sentence-based mixcase" src="https://github.com/Genaios/TextMachina/blob/main/assets/tasks/mixcase_sentences.png?raw=true">
    <figcaption>Example from a mixcase task (tagging), interleaving generated sentences with human texts.</figcaption>
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
  <picture>
    <img alt="CLI interface showing generated and human text for word-span-based mixcase" src="https://github.com/Genaios/TextMachina/blob/main/assets/tasks/mixcase_wordspans.png?raw=true">
    <figcaption>Example from a mixcase task (tagging), interleaving generated word spans with human texts.</figcaption>
  </picture>
  <br/>
  <br/>
</p>

However, the users can build datasets for other tasks not included in ![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true) just by leveraging the provided task types. For instance, datasets for mixcase classification can be built from datasets for mixcase tagging, or datasets for mixcase attribution can be built using the generation model name as label.