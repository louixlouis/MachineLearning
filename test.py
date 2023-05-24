import torch

'''
intrinsic - 내제가치
extrinsic - 외적
'''

'''
torch.sum(input, dim, keepdim)
'''
x = torch.randn(2, 3, 4, 5)
print(x)
print(torch.sum(x, -1, keepdim=True))
print(torch.cumsum(torch.sum(x, -1, keepdim=True), -1))

'''
contiguous
tensor의 shape변화에 따른 저장된 메모리 순서 조정
'''

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

import numpy as np

list_1D = [1, 2, 3, 4]
x = np.array(list_1D); print(x)
x = np.array(list_1D, dtype=int); print(x)
x = np.array(list_1D, dtype=float); print(x)
x = np.array(list_1D, dtype=complex); print(x)

print('-'*20)
arr1 = np.array([1,2,3])
arr2 = np.array([2,3,1])
print(arr1+arr2)
print(arr1+2)
print(arr1-arr2)
print(arr1-1)
print(arr1*arr2)
print(arr1*5)
print(arr1/arr2)
print(arr1/10)

print('-'*20)
arr1 = [1, 2, 3]
arr2 = [2, 3, 1]
print(arr1+arr2)
# print(arr1+2)
# print(arr1-arr2)
# print(arr1-1)
# print(arr1*arr2)
print(arr1*2)
# print(arr1/arr2)
# print(arr1/10)

print('-'*20)
L1 = [[1, 2, 3], [4, 5, 6]]
arr1 = np.array(L1)
print(L1[0])
print(L1[0][1])

print(arr1[0])
print(arr1[0][1])
print(arr1[0, 1])

print(arr1[0:1])
print(arr1[0:1,1:3])
print('-'*20)
print(L1[0])
print(L1[0][1])
print(L1[0:1])
print(L1[0:1][0][1:3])
print('-'*20)
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(np.concatenate((arr1, arr2)))
print(arr1.sum())
print(arr1.prod())
print(arr1.mean())
print(arr1.std())