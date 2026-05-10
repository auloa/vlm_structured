from PIL import Image
from vlm.models.vision_encoder import DonutVisionEncoder
from vlm.utils.device import get_device
from vlm.utils.hf_login import login_to_hf

login_to_hf()

img = Image.new('RGB', (224, 224), color = (128, 100, 100))

encoder = DonutVisionEncoder(model_name="microsoft/dit-base", device = get_device())
output = encoder([img])

print('hidden_size:', encoder.hidden_size)
print('output shape:', output.shape)
print('dtype:', output.dtype)
print('device:', output.device)


trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)

print(f"Trainable parameters: {trainable}")
print(f"Frozen parameters: {frozen}")

assert trainable == 0, 'vision encoder is NOT frozen!'
print('frozen check passed')