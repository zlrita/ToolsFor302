# 加载数据部分

## 加载mhd格式数据

 load_moqui_dose(mhd_file_path):

    """

    读取 .mhd 文件，并返回对应的剂量图像数组。

    Args:

        mhd_file_path(str): 包含剂量图像的文件路径。

    Returns:

        dose_image_array(numpy.ndarray): 剂量图像数组。

    """

## 加载一个文件夹下的所有mhd数据并求和

load_folder_dose(folder_path):

    """

    读取文件夹下所有 .mhd 文件，并返回对应的剂量图像数组。

    Args:

        folder_path(str): 包含 .mhd 文件的文件夹路径。

    Returns:

        dose_image_array(numpy.ndarray): 剂量图像数组。

    """

## 加载bin格式的数据

load_bin_dose(bin_file_path, type):

    """

    读取 .bin 文件，并返回对应的剂量图像数组。

    Args:

        bin_file_path(str): 包含剂量图像的文件路径。

        type(str): 剂量图像数据类型, 包括 'float', 'double', 'int'。

    Returns:

        bin_dose(numpy.ndarray): 剂量图像数组, 返回的是一维数组, 需要自己reshape形状。

    """

# 可视化部分

## 显示两个数据叠加的单个切片

show_double_dose_slice(dose1, dose2, slice_num):

    """

    显示两个剂量分布的叠加切片。

    Args:

        dose1(numpy.ndarray): 第一个剂量分布。

        dose2(numpy.ndarray): 第二个剂量分布。

        slice_num(int): 切片索引, 显示当前要显示的切片。

    """ 

效果图如下

（暂时没得，找不到我的CT数据了）

## 显示单个数据的单个切片

show_single_dose_slice(dose, slice_num, dose_min, dose_max):

    """

    显示单个剂量分布的切片。

    Args:

        dose(numpy.ndarray): 剂量分布。

        slice_num(int): 切片索引, 显示当前要显示的切片。

        dose_min(double): 剂量最小值, 用于控制colorBar的显示范围。

        dose_max(double): 剂量最大值, 用于控制colorBar的显示范围。

    """

效果图如下

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2025-02-27-16-30-12-image.png)

## 显示单个数据的切片，可以交互切换切片

show_single_slice_interactive(dose):

    """

    显示单个剂量分布的切片, 交互式版本。

    Args:

        dose(numpy.ndarray): 要显示的剂量分布。

    """

效果图如下

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2025-02-27-16-26-10-image.png)

## 显示两个Dose的射程图

show_range(dose1, dose2, x_idx, y_start, y_end, z_idx):

    """

    显示两个剂量分布的射程图。

    Args:

        dose1(numpy.ndarray): 第一个剂量分布。

        dose2(numpy.ndarray): 第二个剂量分布。

        x_idx(int): 显示的x方向的索引。

        y_start(int): 显示的y方向的起始索引。

        y_end(int): 显示的y方向的结束索引。

        z_idx(int): 显示的z方向的索引。

    """

效果图如下

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2025-02-27-16-25-32-image.png)

## 显示两个Dose的Line Dose图

show_line_dose(dose1, dose2, x_start, x_end, y_idx, z_idx):

    """

    显示line dose, 横向的拉一条线, 这条线上的dose分布

    Args:

        dose1(numpy.ndarray): 第一个剂量分布。

        dose2(numpy.ndarray): 第二个剂量分布。

        x_start(int): 显示的x方向的起始索引。

        x_end(int): 显示的x方向的结束索引。

        y_idx(int): 显示的y方向的索引。

        z_idx(int): 显示的z方向的索引。

    """

效果图如下

![](C:\Users\Administrator\AppData\Roaming\marktext\images\2025-02-27-16-26-28-image.png)

# 数学部分

## 创建一个Mask矩阵

create_cubeMask(shape, edge_size, mask_value):

    """

    创建一个cube mask, 在shape大小的矩阵内, 正中心的位置生成一个边长为edge_size，值为mask_value的mask。

    Args:

        shape(tuple): 矩阵的大小。

        edge_size(int): 边长大小。

        mask_value(double): 填充的数值。

    Returns:

        mask(numpy.ndarray): 填充了mask_value的矩阵。

    """

## 计算Gamma通过率

cal_gamma(dose_ref, dose_eval, dose_threshold, distance_threshold, dose_cutoff, interp_fraction, ram_available):

    """

    计算Gamma通过率

    Args:

        dose_ref (numpy.ndarray): 第一个剂量分布。

        dose_eval (numpy.ndarray): 第二个剂量分布。

        dose_threshold (double): 剂量差异百分比阈值

        distance_threshold (double): 空间距离阈值 (mm)

        dose_cutoff (double): 剂量截断百分比

        interp_fraction (int): 插值精度, 建议至少10

        ram_available (int): 可用内存（单位字节）

    """
