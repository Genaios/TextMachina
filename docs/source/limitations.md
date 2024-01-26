⚠️ Limitations
========

- Although you can use any kind of extractor to build boundary detection datasets, it is highly recommended to use the *sentence_prefix* or
*word_prefix* extractors with a random number of sentences/words to avoid biases that lead boundary detection models to just count sentences or words.

- ![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true) attempts to remove disclosure patterns (e.g., "*As an AI language model ...*") with a limited set of regular expressions, but they depend on the LLM and the language. We strictly recommend to first *explore* your dataset looking for these biases, and modify the postprocessing or the prompt template accordingly to remove them.

- Generating multilingual datasets is not well supported yet. At this moment, we recommend to generate independent datasets for each language and combine them together out of ![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true).

- Generating machine-generated code datasets is not well supported yet.