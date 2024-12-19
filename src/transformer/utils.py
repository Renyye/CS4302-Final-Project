

def save_tensor_to_file(tensor, filename, name):
    # 保存张量的形状和数据到文件
    with open(filename, 'a') as f:
        f.write(f"Shape: {tensor.shape}, type: {type(tensor)} \n")
        f.write(f"Name: {name}, dtype: {tensor.dtype}\n")
        f.write(f"Data: {tensor.cpu().detach().numpy()}\n")
        f.write("\n" + "="*50 + "\n")