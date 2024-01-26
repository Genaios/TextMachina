💾 Caching
========

![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true) TextMachina caches each dataset it generates through the CLI endpoints under a run name. 
The specific run name is given as the last message in the logs, and can be used with `--run-name <run-name>` to continue from interrupted runs.
The default cache dir used by ![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true) TextMachina is `/tmp/text_machina_cache`. 
It can be modified by setting `TEXT_MACHINA_CACHE_DIR` to a different path.
