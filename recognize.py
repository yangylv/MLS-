import numpy as np
import MLS_generate
import grayscale
from PIL import Image
from scipy.optimize import curve_fit
import cv2
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import original

def find_subarray(sub, original):
    # 获取 sub 和 original 的维度
    sub = np.array(sub);  sub = sub.astype(int)
    sub_shape = sub.shape
    original = original.astype(int)
    original_shape = original.shape
    
    sub_rows, sub_cols = sub_shape[:2]
    original_rows, original_cols = original_shape[:2]
    
    # 遍历 original 中的每个可能的起始点
    for i in range(original_rows - sub_rows + 1):
        for j in range(original_cols - sub_cols + 1):
            # 检查从 (i, j) 开始的子数组是否与 sub 匹配
            if np.array_equal(sub, original[i:i + sub_rows, j:j + sub_cols]):
                return (i, j)
    
    # 如果没有找到，返回 None
    return None

#按周期进行分割，分割成一个个条状
def split_image_and_read_pixels(image_path, P_ref, d):
    image = Image.open(image_path)
    width, height = image.size

    pixel_data = []

    start_x = round(d)
    P = round(P_ref)

    # 进行竖条分割
    for x in range(start_x, width, P):
        # 计算每个竖条的右边界
        right_x = min(x + P, width)
        
        # 如果竖条的宽度小于 P，则舍弃该竖条
        if (right_x - x) < P:
            break
            
        # 裁剪出竖条
        box = (x, 0, right_x, height)
        strip = image.crop(box)
        
        # 获取竖条的像素数据
        pixels = list(strip.getdata())
        pixel_array = [pixels[i * (right_x - x):(i + 1) * (right_x - x)] for i in range(height)]
        pixel_data.append(pixel_array)
    
    pixel_data = np.array(pixel_data)
    pixel_data = np.swapaxes(pixel_data, 1,2)
        

    return pixel_data

def horizontal_split(ver_split_data, popt, circle_size=0.4, spacing = 1.2, bound_w = [0, np.inf]):
    
    array_detected = []
    d_y = []
    ratio_list_full = []
    
    for i in range(ver_split_data.shape[0]):
        pixel_col = ver_split_data[i]
        gray_vertical = grayscale.gray_scale(pixel_col)

        gray_vertical = np.array(gray_vertical, dtype=np.uint8)
        vertical_filter = median_filter(gray_vertical, size=2)
        vertical_filter = gray_vertical
        _, binary_vertical = cv2.threshold(vertical_filter, 250, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #binary_vertical = gray_vertical
        

        sum_row = np.sum(binary_vertical, axis=0)
        normalized_sum = sum_row/np.mean(sum_row)


        x = np.arange(normalized_sum.size)
        def gx_fixed(x, phi):
            return grayscale.gx(x, *popt[:3], phi)

        w_min = bound_w[0]; w_max = bound_w[1]
        bounds = ([0, 0, w_min, -2*np.pi], [2, 2, w_max, 0])
        fitting_params, pcov = curve_fit(grayscale.gx, x, normalized_sum, bounds=bounds)

        #[phi_fitted], pcov = curve_fit(gx_fixed, x, normalized_sum, bounds=[-2*np.pi, 0])
        
        p = round(2*np.pi/fitting_params[2])
        phi_fitted = fitting_params[3]
        d_float = -phi_fitted/fitting_params[2]
        d = round(-phi_fitted/fitting_params[2])
       

        d_y.append(d_float)



        if d < 0 or d >= gray_vertical.shape[1]:
            raise ValueError("d must be within the valid range of columns.")
        
        array_split = gray_vertical[:,d:]
        num_rows = array_split.shape[1] // p
        split_hor = np.array(np.split(array_split[: , :num_rows * p], num_rows, axis=1))

    
        array_hor_test = []
        ratio_list = []
        for j in range(split_hor.shape[0]):

            single_circle = split_hor[j,:,:]

            ratio = black_ratio(single_circle)
            

            array_hor_test.append(ratio_judge(ratio, circle_size, spacing))

            ratio_list.append(ratio)


        array_detected.append(array_hor_test)

        ratio_list_full.append(ratio_list)
    #print(ratio_list_full)


    #array_detected = np.array(array_detected)
    #print("array")
    #print(array_detected)


    distance_y = np.mean(d_y)

    return distance_y, array_detected
    return distance_y, np.swapaxes(array_detected, 0, 1).tolist()



#计算黑色像素占比
def black_ratio(pixel_array):
    white_pixel_count = np.sum(pixel_array>200)
    total_pixel_count = pixel_array.size
    black_pixel_count = total_pixel_count-white_pixel_count
    black_pixel_count = np.sum(pixel_array<100)
    ratio = black_pixel_count/total_pixel_count
    #ratio_w = white_pixel_count/total_pixel_count
    return ratio

def ratio_judge(n, circle_size=0.4, spacing=1.2):
    rmax = circle_size; r1 = circle_size*0.867; r2 = circle_size*0.501
    a_square = spacing**2
    n0 = np.pi*(rmax**2-r1**2)/a_square
    n1 = np.pi*(rmax**2-r1**2+r2**2)/a_square
    n2 = np.pi*(rmax**2-r2**2)/a_square
    n3 = np.pi*rmax**2/a_square

    """
    print("n0=", n0)
    print("n1=", n1)
    print("n2=", n2)
    print("n3=", n3)
    """

    dn = 0.04
    if n>(n3-0.06):
        return [1,1]
    elif (n>(n2-0.06) and n<(n2+0.02)):
        return [0,1]
    elif (n<(n1+0.25) and n>(n1-dn)):
        return [1,0]
    elif (n<(n0+0.02)):
        return [0,0]



def measure(image_test, popt):
    p_ref = grayscale.get_p_test(image_test, bound_w)

    phi = grayscale.compare(image_test, popt)
    distance_x = -phi[0]/popt[2]
    print("P=", p_ref)
    print("d=", distance_x)
    #测试图片的序列数

    ver_split = split_image_and_read_pixels(image_test, p_ref, distance_x)
    print("P_x=", p_ref)
    print("distance x=", distance_x)
    distance_y, hor_split = horizontal_split(ver_split, popt, circle_size, spacing, bound_w)
    #print(np.array(hor_split).shape)

    
    position = find_subarray(hor_split, whole_array)
    if position is not None:
        (x, y) = position
        print("序列的移动位数为x=",x,"y=",y)
    else:
        print(whole_array[4:10,4:10].tolist())
    print("x方向移动距离: ",distance_x, "y方向移动距离: ",distance_y)
    return distance_x, distance_y, x, y

def ref_measure(ref, popt):
    p_ref = 2*np.pi/popt[2]

    phi = popt[3]
    distance_x = -phi/popt[2]
    print("P=", p_ref)
    print("d=", distance_x)
    

    ver_split = split_image_and_read_pixels(ref, p_ref, distance_x)
    print("P_x=", p_ref)
    print("distance x=", distance_x)
    distance_y, hor_split = horizontal_split(ver_split, popt, circle_size, spacing, bound_w)
    #print(np.array(hor_split).shape)

    print("x方向移动距离: ",distance_x, "y方向移动距离: ",distance_y)
    position = find_subarray(hor_split, whole_array)
    if position is not None:
        (x, y) = position
        print("序列的移动位数为x=",x,"y=",y)
    else:
        print("没有找到x和y")
    
    return distance_x, distance_y, x, y




if __name__ == "__main__":
    
    #生成原始完整的数组
    
    whole_array = original.get_whole_array()
    whole_array = whole_array.astype(int)

    #记录圆圈的半径和间隔
    circle_size = 0.4
    spacing = 1.2


    #根据ref图片获得周期
    bound_w = [0.2, 0.35]
    popt = grayscale.reference("ref.png", bound_w)

    print("参考图片测量")
    dx_ref, dy_ref, x_ref, y_ref = ref_measure("ref.png", popt)
    
    #输入测试图片
    print("测试图片测量")
    image_test = "test2.png"
    dx_test, dy_test, x_test, y_test = measure(image_test, popt)



    dx = dx_test-dx_ref; dy = dy_test-dy_ref
    xmove = x_test - x_ref; ymove = y_test-y_ref
    print("相对于参考图片的序列移动为x=",xmove,"y=",ymove,"相位移动距离phi x=", dx, "phi y=", dy)
    







    #print(whole_array.shape)
    cut = whole_array[7:14,3:9]
    #print(cut)
    print("new")
    #print(np.array(hor_split))
    
    