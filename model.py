import tensorflow as tf
import func


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG

    # include_top=False：不包括vgg19的全连接层
    # weights='imagenet'：加载在ImageNet数据集上预训练的权重
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # 在训练过程中冻结vgg19的权重
    vgg.trainable = False
    # 列表推导式：遍历vgg19模型的layer_names，把vgg.get_layer(name).output应用到每一个遍历出的因素name上
    # 结果列表为layer_names中每个层名对应的输出张量
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


# StyleContentModel类继承了tf.keras.models.Model，用于创建一个包含风格和内容层的模型
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        # 传入风格层和内容层的模型，构建vgg模型
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    # call方法是一个处理流程，它将输入图像转换为VGG19模型的特征表示，然后分割这些特征为风格和内容部分，
    # 并计算风格特征的Gram矩阵，最后将这些信息组织成易于访问的字典形式
    def call(self, inputs):
        # "Expects float input in [0,1]"
        # 将输入图像的像素值从【0，1】范围扩展到[0,255]范围
        inputs = inputs * 255.0
        # 应用vgg19的预处理
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])
        # 对于风格输出的每个特征张量，都使用gram_matrix函数计算其gram矩阵
        style_outputs = [func.gram_matrix(style_output)
                         for style_output in style_outputs]
        # 字典推导式，键——content_name，值——value
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        # 返回字典，元素键值对中的值也是字典
        return {'content': content_dict, 'style': style_dict}