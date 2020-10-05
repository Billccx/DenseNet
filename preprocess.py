def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

filepath='D:/python/cifar10/cifar-10-batches-py/data_batch_1'
dic=unpickle(filepath)
print(dic.keys())