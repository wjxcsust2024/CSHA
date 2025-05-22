'''
输入：B, HxW, C
输出：B, C, H, W
'''
def dim3_4(input):
    B, N, C = input.shape
    H = W = int(N**0.5)
    return input.reshape(B, H, W, C).permute(0, 3, 1, 2)

'''
输入：B, C, H, W
输出：B, HxW, C
'''
def dim4_3(input):
    B, C, H, W = input.shape
    return input.reshape(B, C, -1).permute(0, 2, 1)

'''
输入：B, H, W, C
输出：B, HxW, C
'''
def dim4_3_mamba(input):
    B, H, W, C = input.shape
    return input.reshape(B, H*W, C)

'''
输入：B, HxW, C
输出：B, H, W, C
'''
def dim3_4_mamba(input):
    B, N, C = input.shape
    H = W = int(N**0.5)
    return input.reshape(B, H, W, C)
