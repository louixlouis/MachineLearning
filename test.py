import torch

# torch.sum(input, dim, keepdim)
x = torch.randn(2, 3, 4, 5)
print(x)
print(torch.sum(x, -1, keepdim=True))
print(torch.cumsum(torch.sum(x, -1, keepdim=True), -1))

# contiguous
# tensor의 shape변화에 따른 저장된 메모리 순서 조정
'''
Related methods
narrow(), view(), expand(), transpose(), ...

stride() : 데이터 저장 방향
is_contiguous : whether contiguous is True or not
'''

'''
torch.searchsorted(sorted_sequence, value, right)
right : False - first one, True - last one
'''