🔄 Common Use Cases
========

There is a set of common use cases with ![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true). Here's how to carry them out using the *explore* and *generate* endpoints.

| Use case                                                                    | Command                                                                                                                       |
|-----------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| Explore a dataset of 10 samples for MGT detection and show metrics          | <pre>text-machina explore \ <br>--config-path config.yaml \ <br>--task-type detection \ <br>--max-generations 10 \ <br>--metrics_path metrics.yaml</pre>  |
| Explore an existing dataset for MGT detection and show metrics              | <pre>text-machina explore \ <br>--config-path config.yaml \ <br>--run-name greedy-bear \ <br>--task-type detection \ <br>--metrics_path metrics.yaml</pre> |
| Generate a dataset for MGT detection                                        | <pre>text-machina generate \ <br>--config-path config.yaml \ <br>--task-type detection</pre>                                                       |
| Generate a dataset for MGT attribution                                      | <pre>text-machina generate \ <br>--config-path config.yaml \ <br>--task-type attribution</pre>                                                     |
| Generate a dataset for boundary detection                                   | <pre>text-machina generate \ <br>--config-path config.yaml \ <br>--task-type boundary</pre>                                                        |
| Generate a dataset for mixcase detection                                | <pre>text-machina generate \ <br>--config-path config.yaml \ <br>--task-type mixcase</pre>                                                        |
| Generate a dataset for MGT detection using config files in a directory tree | <pre>text-machina generate \ <br>--config-path configs/ \ <br>--task-type detection</pre>                                                          |