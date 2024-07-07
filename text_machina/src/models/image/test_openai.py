from ...config import ModelConfig
from .openai import OpenAIModel

model_config = ModelConfig(
    provider="openai",
    model_name="dall-e-3",
    max_retries=5,
    threads=8,
)
generation_config = {"n": 1, "size": "1024x1024", "quality": "standard"}

model = OpenAIModel(model_config)
prompt = "Context: The bathing machine was a device, popular from the 18th century until the early 20th century, to allow people to change out of their usual clothes, change into swimwear, and wade in the ocean at beaches. Bathing machines were roofed and walled wooden carts rolled into the sea. Some had solid wooden walls, others canvas walls over a wooden frame, and commonly walls at the sides and curtained doors at each end. The use of bathing machines as part of the etiquette for sea-bathing was more rigorously enforced upon women than men, but it was to be observed by both sexes among those who wished to behave respectably. Especially in Britain, men and women were usually segregated, so that people of the opposite sex should not see them in their bathing suits, which were not considered proper clothing in which to be seen in public. Image: 'Sea bathing in mid Wales c.1800. Several bathing machines can be seen'"
for image in model.batched_generate([prompt], generation_config):
    print(image)
