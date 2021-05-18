import torch

probs = torch.Tensor([.3,0.3,0.3])
desired_probs = torch.Tensor([1/3,1/3,1/3])
print(torch.log(probs) )
loss = torch.nn.functional.kl_div(torch.log(probs), desired_probs)
print(loss)
print(-torch.log(loss).item())