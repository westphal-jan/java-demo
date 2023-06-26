import torch
from torch import nn

class DummyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(x)

torch.manual_seed(0)

model = DummyModel()
script_model = torch.jit.script(model)
script_model.save("dummy-model.pt")

x = torch.tensor([[1., 2., 3., 4., 5.]])

print("Unquantized model:")
print(torch.jit.load("dummy-model.pt")(x).tolist())

q_dict = {nn.Linear: torch.ao.quantization.default_dynamic_qconfig}

quantized_model = torch.quantization.quantize_dynamic(model, q_dict)
quantized_script_model = torch.jit.script(quantized_model)
quantized_script_model.save("dummy-model-quantized.pt")

print("Quantized model:")
print(torch.jit.load("dummy-model-quantized.pt")(x).tolist())
