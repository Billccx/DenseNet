import matplotlib.pyplot as plt
epoch=[x for x in range(1,5)]
trainacc=[0.1,0.5,0.9,0.9]
testacc=[0.1,0.2,0.4,0.6]
plt.plot(epoch,trainacc,color='red',label='train')
plt.plot(epoch,testacc,color='blue',label='test')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.title('Accruacy')
plt.legend()
plt.show()
plt.savefig('result.jpg')


'''
import matplotlib.pyplot as plt

epochs = [0, 1, 2, 3]
acc = [4, 8, 6, 5]
loss = [3, 2, 1, 4]

plt.plot(epochs, acc, color='r', label='acc')  # r表示红色
plt.plot(epochs, loss, color=(0, 0, 0), label='loss')  # 也可以用RGB值表示颜色

#####非必须内容#########
plt.xlabel('epochs')  # x轴表示
plt.ylabel('y label')  # y轴表示
plt.title("chart")  # 图标标题表示
plt.legend()  # 每条折线的label显示
#######################
plt.savefig('test.jpg')  # 保存图片，路径名为test.jpg
plt.show()  # 显示图片
'''