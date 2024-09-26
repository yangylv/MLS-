import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
import cv2
from scipy.ndimage import median_filter

def get_pixel(image_file):
    image = Image.open(image_file)
    width, height = image.size
    
    rgb_image = image#.convert("RGB")
    pixels = list(rgb_image.getdata())
    pixel_array = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return np.swapaxes(np.array(pixel_array), 0, 1), width, height

def gray_scale(pixel_array):

    gray_pixel = np.ones((pixel_array.shape[0], pixel_array.shape[1]))
    R = pixel_array[:,:,0]; G = pixel_array[:,:,1]; B = pixel_array[:,:,2]
    sum_all = np.sum(R)+np.sum(G)+np.sum(B)
    wr = np.sum(R)/sum_all; wg = np.sum(G)/sum_all; wb = np.sum(B)/sum_all
    for i in range(pixel_array.shape[0]):
        for j in range(pixel_array.shape[1]):
            r = pixel_array[i][j][0]
            g = pixel_array[i][j][1]
            b = pixel_array[i][j][2]
            gray_pixel[i][j] = wr*r + wg* g + wb*b
    return gray_pixel

def col_sum(array):
    sum_arr = np.sum(array, axis=1)
    normalized_sum = sum_arr/np.mean(sum_arr)
    return normalized_sum

def trim_unchanged_segments(arr):
    if len(arr) < 2:
        return arr  # 如果数组长度小于2，直接返回
    
    # 找到变化的起点
    start = 0
    while start < len(arr) - 1 and arr[start] == arr[start + 1]:
        start += 1

    # 如果整个数组都是相同的值，直接返回空数组
    if start == len(arr) - 1:
        return []

    # 找到变化的终点
    end = len(arr) - 1
    while end > 0 and arr[end] == arr[end - 1]:
        end -= 1

    # 返回变化部分
    return arr[start + 1:end]

gx = lambda x,a,b,w,c : a+b*np.sin(w*x+c)
def fitting_sin(x, y, bound_w=[0, np.inf]):
    w_min = bound_w[0]; w_max = bound_w[1]
    bounds = ([0.1, 0.1, w_min, -2*np.pi], [2, 2, w_max, 0])
    popt, pcov = curve_fit(gx, x, y, bounds=bounds)
    return popt


#参考平面的拟合
def reference(imagefile, bound_w = [-np.inf, np.inf]):
    pixel_array, width, height = get_pixel(imagefile)
    gray_pixel = gray_scale(pixel_array)
    gray_pixel = np.array(gray_pixel, dtype=np.uint8)
    pixel_filter = median_filter(gray_pixel, size=3)
    _, binary_array = cv2.threshold(pixel_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sum_arr_ori = col_sum(binary_array)
    sum_arr = sum_arr_ori
    #sum_arr = trim_unchanged_segments(sum_arr_ori)
    x = np.arange(sum_arr.size)
    
    popt = fitting_sin(x, sum_arr, bound_w)
    plt.scatter(x, sum_arr, s=3)
    plt.plot(x, gx(x, *popt), color="C1")
    plt.show()
    print(popt)
    return popt


#测试面的拟合和对比
def compare(image_new, popt):
    '''
    输出phi_fitted和phi_difference
    '''
    pixel_array, width, height = get_pixel(image_new)
    gray_pixel = gray_scale(pixel_array)
    gray_pixel = np.array(gray_pixel, dtype=np.uint8)
    pixel_filter = median_filter(gray_pixel, size=3)
    _, binary_array = cv2.threshold(pixel_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sum_arr_ori = col_sum(binary_array)
    sum_arr = sum_arr_ori
    #sum_arr = trim_unchanged_segments(sum_arr_ori)
    x = np.arange(sum_arr.size)

    def gx_fixed(x, phi):
        return gx(x, *popt[:3], phi)

    [phi_fitted], pcov = curve_fit(gx_fixed, x, sum_arr, bounds=[-2*np.pi, 0])

    plt.scatter(x, sum_arr, s=3)
    plt.plot(x, gx(x, *popt), color="C1", label = "reference")
    plt.plot(x, gx(x, *popt[:3], phi_fitted), color="C2", label = "test")
    plt.legend()
    plt.show()

    phi_difference = phi_fitted - popt[3]
    return phi_fitted, phi_difference



def get_p_test(image_test, bound_w = [0, np.inf]):
    pixel_array, width, height = get_pixel(image_test)
    gray_pixel = gray_scale(pixel_array)
    gray_pixel = np.array(gray_pixel, dtype=np.uint8)
    pixel_filter = median_filter(gray_pixel, size=3)
    _, binary_array = cv2.threshold(pixel_filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sum_arr_ori = col_sum(binary_array)
    sum_arr = sum_arr_ori
    #sum_arr = trim_unchanged_segments(sum_arr_ori)
    x = np.arange(sum_arr.size)
    
    popt = fitting_sin(x, sum_arr, bound_w)
    
    p = 2*np.pi/popt[2]
    return p



if __name__ == "__main__":

    bound_w = [0.2, 0.35]   #上界不能超过2w


    popt = reference("ref.png", bound_w)
    phi_fitted, phi_difference = compare("test.png", popt)
    print(phi_fitted, phi_difference)
    
