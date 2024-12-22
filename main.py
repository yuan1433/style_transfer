import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
# 用于显示进度条
from tqdm import trange
import numpy as np
import time
import func
import model
from config import *

# ui设计需要
import tkinter as tk
from tkinter import filedialog,messagebox
import shutil
import os

# class内的各方法内的self.成员是共享的
class UI:

    def __init__(self,root):
        self.root=root
        self.root.title("图像风格迁移")

        self.content_select_label=tk.Label(root,text="选择内容文件")
        self.content_select_label.config(font=("Arial", 16))
        self.content_select_label.place(x=10, y=10, anchor=tk.NW)

        self.content_fileName_var=tk.StringVar()
        self.content_fileName_entry=tk.Entry(root,textvariable=self.content_fileName_var,width=50,state='readonly')
        self.content_fileName_entry.place(x=150,y=20,anchor=tk.NW)

        self.content_select_button=tk.Button(root,text="选择",command=lambda: self.select_file_wrapper(self.content_fileName_entry))
        self.content_select_button.config(font=("Arial", 16))
        self.content_select_button.place(x=520,y=10,anchor=tk.NW)

        self.style_select_label = tk.Label(root, text="选择风格文件")
        self.style_select_label.config(font=("Arial", 16))
        self.style_select_label.place(x=10, y=80, anchor=tk.NW)

        self.style_fileName_var = tk.StringVar()
        self.style_fileName_entry = tk.Entry(root, textvariable=self.style_fileName_var, width=50, state='readonly')
        self.style_fileName_entry.place(x=150, y=90, anchor=tk.NW)

        self.style_select_button = tk.Button(root, text="选择", command=lambda: self.select_file_wrapper(self.style_fileName_entry))
        self.style_select_button.config(font=("Arial", 16))
        self.style_select_button.place(x=520, y=80, anchor=tk.NW)

        self.project_dir=os.path.dirname(os.path.abspath(__file__))

        self.style_weight_label = tk.Label(root, text="风格权重")
        self.style_weight_label.config(font=("Arial", 16))
        self.style_weight_label.place(x=10, y=150, anchor=tk.NW)

        # 少了个root时会运行失败：就是刚运行就结束了
        self.style_weight_var = tk.StringVar(root,"0.5")
        self.style_weight_entry = tk.Entry(root, textvariable=self.style_weight_var, width=50)
        self.style_weight_entry.place(x=150, y=150)

        self.content_weight_label = tk.Label(root, text="内容权重")
        self.content_weight_label.config(font=("Arial", 16))
        self.content_weight_label.place(x=10, y=220, anchor=tk.NW)

        self.content_weight_var = tk.StringVar(root, "1e4")
        self.content_weight_entry = tk.Entry(root, textvariable=self.content_weight_var, width=50)
        self.content_weight_entry.place(x=150, y=220)

        self.epochs_label = tk.Label(root, text="训练轮数")
        self.epochs_label.config(font=("Arial", 16))
        self.epochs_label.place(x=10, y=290, anchor=tk.NW)

        self.epochs_var = tk.StringVar(root, "10")
        self.epochs_entry = tk.Entry(root, textvariable=self.epochs_var, width=50)
        self.epochs_entry.place(x=150, y=290)

        self.steps_per_epoch_label = tk.Label(root, text="每轮迭代次数")
        self.steps_per_epoch_label.config(font=("Arial", 16))
        self.steps_per_epoch_label.place(x=10, y=360, anchor=tk.NW)

        self.steps_per_epoch_var = tk.StringVar(root, "100")
        self.steps_per_epoch_entry = tk.Entry(root, textvariable=self.steps_per_epoch_var, width=50)
        self.steps_per_epoch_entry.place(x=150, y=360)

        # self.confirm_style_weight_button = tk.Button(root,text="确定", command=lambda: self.update_on_confirm(self.style_weight_entry,self.style_weight_var))
        # self.confirm_style_weight_button.place(x=520,y=150,anchor=tk.NW)

        self.confirm_button=tk.Button(root,text="开始生成",command=self.style_transfer)
        self.confirm_button.config(font=("Arial", 16))
        self.confirm_button.place(x=520, y=420, anchor=tk.NW)

    def select_file(self,entry):
        # 打开文件选择对话框
        file_path=filedialog.askopenfilename()
        if file_path:
            # 提取文件名和后缀
            file_name=os.path.basename(file_path)
            # 更新文本框内容
            entry.config(state='normal')  # 允许编辑以设置新值（尽管这里设置为只读，但为更新文本而临时允许）
            entry.delete(0, tk.END)  # 删除现有内容
            entry.insert(0, file_name)  # 插入新文件名
            entry.config(state='readonly')  # 恢复只读状态
            # 文件复制到项目目录下
            save_file_path=os.path.join(self.project_dir,file_name)
            try:
                # 检查文件是否已存在，避免覆盖
                if not os.path.exists(save_file_path):
                    shutil.copy2(file_path,save_file_path)
                    messagebox.showinfo("成功",f"文件已成功保存到项目目录下")
                else:
                    messagebox.showwarning("警告",f"文件{file_name}已存在于项目目录，不再进行复制")
            except Exception as e:
                messagebox.showerror("错误",f"保存文件时出错：{e}")

    # 这是一个包装函数，用于与Tkinter的command属性兼容
    # 它接收按钮点击事件（尽管未使用），并调用select_file方法
    def select_file_wrapper(self, entry):
        self.select_file(entry)

    def style_transfer(self):
        # 设置图像大小
        mpl.rcParams['figure.figsize'] = (12, 12)
        # 不显示网格
        mpl.rcParams['axes.grid'] = False
        content_path=self.content_fileName_entry.get()
        style_path=self.style_fileName_entry.get()

        content_image = func.load_img(content_path)
        style_image = func.load_img(style_path)

        # 定义内容层列表，用于提取内容图像的高级特征
        content_layers = ['block5_conv1']
        # 风格层列表，包括从低级到高级的不同层，以捕捉风格图像中的纹理和模式
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1',
                        'block4_conv1',
                        'block5_conv1']

        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)

        # 提取器
        extractor = model.StyleContentModel(style_layers, content_layers)
        # ['style']键用于从返回的字典中获取风格特征
        # 这些特征以gram矩阵的形式来表示，用于后续计算风格的损失
        # obj(param)，自动调用了子类StyleContentModel重写的call方法
        # extractor(style_image)['style']，调用了call方法后，从返回的字典中获取键为'style'的值
        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        def style_content_loss(outputs):
            # 用来表示style信息的网络层的输出，这里已经计算过Gram矩阵了
            style_outputs = outputs['style']
            # 用来表示content信息的网络层的输出，内容信息不需要计算Gram
            content_outputs = outputs['content']
            # 计算风格损失，对每个风格层计算均方误差并求和
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                                   for name in style_outputs.keys()])
            # 右边为归一化系数，乘以风格权重并除于风格层数，进行归一化
            #
            style_loss *= style_weight / num_style_layers

            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                                     for name in content_outputs.keys()])
            # 右边为归一化系数
            #
            content_loss *= content_weight / num_content_layers
            loss = style_loss + content_loss
            return loss

        opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        image = tf.Variable(content_image + tf.random.truncated_normal(content_image.shape, mean=0.0, stddev=0.08),
                            trainable=True)

        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = style_content_loss(outputs)
                loss += total_variation_weight * func.total_variation_loss(image)

            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(func.clip_0_1(image))

        for n in trange(epochs * steps_per_epoch):
            train_step(image)

        plt.imshow(image.read_value()[0])
        plt.show()
        print(image.read_value()[0].shape)
        Eimg = tf.image.convert_image_dtype(image.read_value()[0], tf.uint8)
        Eimg = tf.image.encode_jpeg(Eimg)
        tf.io.write_file('output.jpg', Eimg)


root=tk.Tk()
root.geometry("700x600")
app=UI(root)
root.mainloop()