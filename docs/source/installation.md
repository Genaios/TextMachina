🔧 Installation
============

**text-machina** supports Python >= 3.8.

## Installing with `pip`

**text-machina** is available [on PyPI](https://pypi.org/project/text-machina/).

You can install all the dependencies with pip:

```bash
pip install text_machina[all]
```

or just with specific dependencies for an specific LLM provider or development dependencies (see `setup.py`)):

```bash
pip install text_machina[anthropic,dev]
```

## Installing from source

To install **text-machina** from source, first clone [the repository](https://github.com/Genaios/TextMachina):

```bash
git clone https://github.com/Genaios/TextMachina
cd TextMachina
```

```bash
pip install .[all]
```

If you're planning to modify the code for specific use cases, you can install ![icon](https://github.com/Genaios/TextMachina/blob/main/assets/typewriter.png?raw=true) in development mode:

```
pip install -e .[dev]
```