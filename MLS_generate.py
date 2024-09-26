# MLS genrate program
import numpy as np
import matplotlib.pyplot as plt

def mls_generate(degree, taps, initial_state, n):
    """
    参数说明：
    degree: 产生MLS序列的级数
    taps：发生XOR运算的位置，可表示为多项式形式
    initial_state：初始数列，不可全为0，且位数需与degree保持一致
    n：产生的序列长度，最大不超过2^dgree-1
    """
    
    if (n<=(2**degree-1)):
        state = initial_state[:]
        seq = []

        for _ in range(n):
            seq.append(state[-1])  # 输出上次循环的最后一位
            # 进行或否运算判断
            feedback = np.bitwise_xor.reduce([state[i] for i in taps])
            state = [feedback] + state[:-1]  # 将feedback放在最前面，整体向后移一位，原来的最后一位作为输出
        return seq
    else:
        print("请保证输入的长度正确")

def get_initial(n):
    arr = np.random.choice([0, 1], size=n)

    while np.all(arr==0):
        arr = np.random.choice([0, 1], size=n)
    return arr.tolist()

    
def generate_2d_mls(rows, colomns, degree, taps):

    n = rows+colomns+3
    initial_state = get_initial(degree)
    mls_1d = mls_generate(degree, taps, initial_state, n)
    x = mls_1d[:rows]; y = mls_1d[-colomns:]
    print(len(x))
    print(len(y))
    mls_2d = np.ones(shape=(rows, colomns, 2))*3
    for i in range(rows):
        for j in range(colomns):
            mls_2d[i][j] = [x[i], y[j]]
    if 3 not in mls_2d:
        return mls_2d
    else:
        print("报错")

def draw_plate(binary_array, circle_size=0.4, spacing=1):
    
    rows, cols, nums = binary_array.shape

    fig, ax = plt.subplots()

    if nums!=2:
        print("报错")
    
    size_middle = circle_size*0.867
    size_small = circle_size*0.501
    linewidth = 0.577
    for i in range(rows):
        for j in range(cols):
            if np.array_equal(binary_array[i, j], [0., 0.]):
                # 大空心圆表示(0, 0)
                circle_large = plt.Circle((j * spacing, (rows - i - 1) * spacing), circle_size,facecolor="black")
                circle_middle = plt.Circle((j * spacing, (rows - i - 1) * spacing), size_middle,facecolor="white",)
                circle_small = plt.Circle((j * spacing, (rows - i - 1) * spacing), size_small, facecolor='white')
            
            
            elif np.array_equal(binary_array[i, j], [0., 1.]):
                # 小圆空心 大圆实心表示(0, 1)
                circle_large = plt.Circle((j * spacing, (rows - i - 1) * spacing), circle_size,facecolor="black")
                circle_middle = None
                circle_small = plt.Circle((j * spacing, (rows - i - 1) * spacing), size_small, facecolor='white')
            
            elif np.array_equal(binary_array[i, j], [1., 0.]):
                # 大圆空心，小圆实心表示(1, 0)
                circle_large = plt.Circle((j * spacing, (rows - i - 1) * spacing), circle_size,facecolor="black", zorder=0.2)
                circle_middle = plt.Circle((j * spacing, (rows - i - 1) * spacing), size_middle,facecolor="white",zorder = 0.5)
                circle_small = plt.Circle((j * spacing, (rows - i - 1) * spacing), size_small, facecolor='black', zorder = 0.8)

            elif np.array_equal(binary_array[i, j], [1., 1.]):
                # 大实心圆表示(1, 1)
                circle_large = plt.Circle((j * spacing, (rows - i - 1) * spacing), circle_size, facecolor='black')
                circle_middle = None
                circle_small = None

            
            
            ax.add_artist(circle_large)
            if circle_small is not None:
                ax.add_artist(circle_small)
            if circle_middle is not None:
                ax.add_artist(circle_middle)
    


    ax.set_xlim(-spacing , cols * spacing)
    ax.set_ylim(-spacing, rows * spacing )
    ax.set_aspect('equal')
    
    # Hide the axes
    plt.axis('off')

    #画一个边框
    '''
    border = plt.Rectangle((-spacing/2,-spacing/2), (cols) * spacing, (rows) * spacing, 
                           edgecolor="black", linewidth=border_width, facecolor='none')
    ax.add_artist(border)
'''
    #plt.savefig("mls.png")
    
    plt.show()

    

    




if __name__ == "__main__":

    #生成一个8阶的MLS
    degree = 8
    taps = [7, 5, 4, 3]
    #initial_state = get_initial(degree)
    initial_state = [1, 0, 1, 1, 0, 1, 1, 0]
    arr = generate_2d_mls(15, 18, degree, taps)

    print(arr.tolist())


    #修改圆圈大小和间隔
    circle_size = 0.4
    spacing = 1.2

    draw_plate(arr, circle_size=circle_size, spacing=spacing)
    
    