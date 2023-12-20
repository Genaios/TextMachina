<!---
Copyright 2023 Genaios

Licensed under the CC BY-NC-ND 4.0 License

You must give appropriate credit, provide a link to the license, and indicate if changes were made.
You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
You may not use the material for commercial purposes.
If you remix, transform, or build upon the material, you may not distribute the modified material.
You are free to copy and redistribute this material as it is in any medium or format
You may obtain a copy of the License at

    https://creativecommons.org/licenses/by-nc-nd/4.0/
-->

<p align="center">
  <picture>
    <img alt="TextMachina" src="assets/title.png" width="352" height="59" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<p align="center">
    <a href="LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-CC_BY_NC_ND_4.0-green">
    </a>
    <a href="">
        <img alt="Documentation" src="https://img.shields.io/badge/Documentation-pending-pink">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0-green">
    </a>
</p>

<h3 align="center">
    <p><b>Unifying strategies to build MGT datasets in a single framework</b></p>
</h3>

![icon](assets/typewriter.png) TextMachina is a modular and extensible Python framework, designed to aid in the creation of high-quality, unbiased datasets to build robust models for MGT-related tasks such as:

- 🔎 **Detection**: detect whether a text has been generated by an LLM.
- 🕵️‍♂️ **Attribution**: identify what LLM has generated a text.
- 🚧 **Boundary detection**: find the boundary between human and generated text.

![icon](assets/typewriter.png) TextMachina provides a user-friendly pipeline that abstracts away the inherent intricacies of building MGT datasets:

- 🦜 **LLM integrations**: easily integrates any LLM provider. Currently, ![icon](assets/typewriter.png) supports LLMs from Anthropic, Cohere, OpenAI, Google Vertex AI, and any model from HuggingFace deployed either locally or remotely through Inference API or Inference Endpoints. See [models](text_machina/src/models/) to implement your own LLM provider.

- ✍️ **Prompt templating**: just write your template with placeholders and let ![icon](assets/typewriter.png) extractors to fill the template and prepare a prompt for an LLM. See [extractors](text_machina/src/extractors) to implement your own extractors.
- 🔒 **Constrained decoding**: automatically infer LLM decoding hyper-parameters from the human texts to improve the quality and reduce the biases of your MGT datasets. See [constrainers](text_machina/src/constrainers) to implement your own constrainers.
- 🛠️ **Post-processing**: post-process functions aimed to improve the quality of any MGT dataset and prevent common biases and artifacts. See [postprocessing](text_machina/src/postprocessing.py) to add new postprocess functions.
- 🌈 **Bias mitigation**: ![icon](assets/typewriter.png) is built with bias prevention in mind and helps you across all the pipeline to prevent introducing spurious correlations in your datasets.
- 📊 **Dataset exploration**: explore the generated datasets and quantify its quality with a set of metrics. See [metrics](text_machina/metrics) and [interactive](text_machina/src/interactive.py) to implement your own metrics and visualizations.

The following diagram depicts the ![icon](assets/typewriter.png)'s pipeline.
<p align="center">
  <picture>
    <img alt="TextMachina Pipeline" src="assets/diagram.png">
  </picture>
  <br/>
  <br/>
</p>

## 🔧 Installation
---

You can install all the dependencies with pip:

```
pip install text_machina[all]
```

or just with specific dependencies for an specific LLM provider or development dependencies (see [setup.py](setup.py)):

```
pip install text_machina[anthropic,dev]
```

You can also install directly from source:

```
pip install .[all]
```

If you're planning to modify the code for specific use cases, you can install ![icon](assets/typewriter.png) in developoment mode:

```
pip install -e .[dev]
```

## 👀 Quick Tour
---

Once installed, you are ready to use ![icon](assets/typewriter.png) for building MGT datasets either using the [CLI](text_machina/src/cli.py) or programmatically.

### 📟 Using the CLI
The first step is to define a YAML configuration file or a directory tree containing YAML files. Look the [examples](etc/examples) folder to be familiar with the configurations.

Then, we can call the *explore* and *generate* endpoints of ![icon](assets/typewriter.png)'s CLI. The *explore* endpoint allows to inspect a small generated dataset using an specific configuration through an interactive interface. For instance, let's suppose we want to check how an MGT detection dataset generated using *[XSum](https://huggingface.co/datasets/EdinburghNLP/xsum)* news articles and *gpt-3.5-turbo-instruct* looks like, and compute some metrics:

```bash
text-machina explore --config-path etc/examples/xsum_gpt-3-5-turbo-instruct_openai.yaml \
--task-type detection \
--metrics-path etc/metrics.yaml \
--max-generations 10
```

<p align="center">
  <picture>
    <img alt="CLI interface showing generated and human text for detection" src="assets/explore.png">
  </picture>
  <br/>
  <br/>
</p>

Great! Our dataset seems to look great, no artifacts, no biases, and high-quality text using this configuration. Let's now generate a whole dataset for MGT detection using that config file. The *generate* endpoint allows you to do that:

```bash
text-machina generate --config-path etc/examples/xsum_gpt-3-5-turbo-instruct_openai.yaml \
--task-type detection
```

A run name will be assigned to your execution and ![icon](assets/typewriter.png) will cache results behind the scenes. If your run is interrupted at any point, you can use `--run-name <run-name>` to recover the progress and continue generating your dataset.

### 👩‍💻 Programmatically

You can also use ![icon](assets/typewriter.png) programmatically. To do that, instantiate a dataset generator by calling *get_generator* with a *Config* object, and run its *generate* method. The *Config* object must contain the input, model, and generation configs, together with the task type for which the MGT dataset will be generated. Let's replicate the previous experiment programmatically:

```python
from text_machina import get_generator
from text_machina import Config, InputConfig, ModelConfig

input_config = InputConfig(
    domain="news",
    language="en",
    quantity=10,
    random_sample_human=True,
    dataset="xsum",
    dataset_text_column="document",
    dataset_params={"split": "test"},
    template=(
        "Write a news article whose summary is '{summary}'"
        "using the entities: {entities}\n\nArticle:"
    ),
    extractor="combined",
    extractors_list=["auxiliary.Auxiliary", "entity_list.EntityList"],
    max_input_tokens=256,
)

model_config = ModelConfig(
    provider="openai",
    model_name="gpt-3.5-turbo-instruct",
    api_type="COMPLETION",
    threads=8,
    max_retries=5,
    timeout=20,
)

generation_config = {"temperature": 0.7, "presence_penalty": 1.0}

config = Config(
    input=input_config,
    model=model_config,
    generation=generation_config,
    task_type="detection",
)
generator = get_generator(config)
dataset = generator.generate()
```

## 🔄 Common Use Cases
---
There is a set of common use cases with ![icon](assets/typewriter.png). Here's how to carry them out using the *explore* and *generate* endpoints.

| Use case                                                                    | Command                                                                                                                       |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Explore a dataset of 10 samples for MGT detection and show metrics          | <pre>text-machina explore \ <br>--config-path config.yaml \ <br>--task-type detection \ <br>--max-generations 10 \ <br>--metrics_path metrics.yaml</pre>  |
| Explore an existing dataset for MGT detection and show metrics              | <pre>text-machina explore \ <br>--config-path config.yaml \ <br>--run-name greedy-bear \ <br>--task-type detection \ <br>--metrics_path metrics.yaml</pre> |
| Generate a dataset for MGT detection                                        | <pre>text-machina generate \ <br>--config-path config.yaml \ <br>--task-type detection</pre>                                                       |
| Generate a dataset for MGT attribution                                      | <pre>text-machina generate \ <br>--config-path config.yaml \ <br>--task-type attribution</pre>                                                     |
| Generate a dataset for boundary detection                                   | <pre>text-machina generate \ <br>--config-path config.yaml \ <br>--task-type boundary</pre>                                                        |
| Generate a dataset for MGT detection using config files in a directory tree | <pre>text-machina generate \ <br>--config-path configs/ \ <br>--task-type detection</pre>                                                          |

## 💾 Caching
![icon](assets/typewriter.png) TextMachina caches each dataset it generates through the CLI endpoints under a run name. 
The specific run name is given as the last message in the logs, and can be used with `--run-name <run-name>` to continue from interrupted runs.
The default cache dir used by ![icon](assets/typewriter.png) TextMachina is `/tmp/text_machina_cache`. 
It can be modified by setting `TEXT_MACHINA_CACHE_DIR` to a different path.


## ⚠️ Notes and Limitations
---

- Although you can use any kind of extractor to build boundary detection datasets, it is highly recommended to use the *sentence_prefix* or
*word_prefix* extractors with a random number of sentences/words to avoid biases that lead boundary detection models to just count sentences or words.

- ![icon](assets/typewriter.png) attempts to remove disclosure patterns (e.g., "*As an AI language model ...*") with a limited set of regular expressions, but they depend on the LLM and the language. We strictly recommend to first *explore* your dataset looking for these biases, and modify the postprocessing or the prompt template accordingly to remove them.

- Span-based MGT datasets are not supported yet.

- Generating multilingual datasets is not well supported yet. At this moment, we recommend to generate independent datasets for each language and combine them together out of ![icon](assets/typewriter.png).

- Generating machine-generated code datasets is not well supported yet.

- LLM providers like Amazon or Azure are not implemented yet.

## 📖 Citation
---
```
@misc{text-machina,
      title={{T}ext{M}achina: {S}eamless {G}eneration of {M}achine-{G}enerated {T}ext {D}atasets}, 
      author={Sarvazyan, Areg Mikael and González, José Ángel and Franco-Salvador, Marc},
      year={2023},
      eprint={TBD},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
## 🤝 Contribute
---

Feel free to contribute to ![icon](assets/typewriter.png) by discussing any issue.

Please install and use the [dev-tools](dev-tools) for correctly formatting the code when contributing to this repo.

## 🏭 Commercial Purposes
---
Please, contact stuart.winter-tear@genaios.ai and marc.franco@genaios.ai if you are interested in using TextMachina for commercial purposes.