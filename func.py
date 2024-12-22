import tensorflow as tf
def load_img(path_to_img):
    # 图像最大边长，与电脑显存大小有关
    max_dim = 720
    # 读取图像文件内容
    img = tf.io.read_file(path_to_img)
    # decode_image 函数解码图像，并指定 channels=3 以确保图像被解码为三通道（RGB）格式
    img = tf.image.decode_image(img, channels=3)
    # float32，大多数深度学习所需格式
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # 宽度、高度的较大值
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    # 在第一个位置添加一个新维度，将图像形状从[height,width,channels]转换成为[1,height,width,channels]
    # 这是为了满足深度学习模型输入的要求，期望接收一个批次的数据
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# 'bijc,bijd->bcd'
# Gran[c, d] = sun_ij (F[batch, i, j, c] * F[batch, i, j, d]) / IJ
# I*J = width * heigh (特征图的
# 第c个特征图和第d个特征图的Gram矩阵值
# 接收一个特征图并返回gram矩阵
def gram_matrix(input_tensor):
    # 完成了内积的计算，即对于每个特征对(c, d)，计算了所有空间位置(i, j)上特征值的乘积之和
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    # 获取输入张量的形状，以便后续计算空间位置的总数
    input_shape = tf.shape(input_tensor)
    # 将空间维度（通常是高度和宽度，在bijc中，input_shape[1]---i, input_shape[2]---j）相乘，得到空间位置的总数num_locations
    # 这里使用tf.cast确保结果是tf.float32类型，以便进行后续的除法运算。
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    # 上一步得到的Gram矩阵除以空间位置的总数num_locations，得到归一化的Gram矩阵
    # 确保Gram矩阵的值不会因为输入张量的空间尺寸而改变
    return result/(num_locations)

# 截断，把值限制到0和1之间
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]

    return x_var, y_var

def total_variation_loss(image):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)
