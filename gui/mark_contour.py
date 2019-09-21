import tkinter
from PIL import Image , ImageTk
import os
from tkinter.filedialog import askdirectory
from multiprocessing import dummy
import numpy as np
import cv2
# from multiprocessing import cpu_count

windows_width = 1024
windows_height = 700

canvas_width = 1024
canvas_height = 512
canvas_position = (0 , 0)

button_pred_width = 20
button_pred_height = 1
button_pred_position = (100 , 600)

button_next_width = 20
button_next_height = 1
button_next_position = (300 , 600)

button_withdraw_width = 20
button_withdraw_height = 1
button_withdraw_position = (500 , 600)

draw_radius = 4

class MainWindow():

    def __init__(self , path , ratios = 2):

        self.sample_path = '%s/t_sample/' % (path , )
        self.contour_path = '%s/t_contour/' % (path ,)
        self.ratios = ratios # 设置图片的放大倍数
        if not os.path.exists(self.contour_path):
            os.makedirs(self.contour_path)

        self.init_list()
        self.init_gui()
        self.is_work = True

        self.init_image(self.index)

    def init_list(self):

        self.img_list = []
        self.contour_list = []
        img_names = os.listdir(self.sample_path)
        img_names = [x for x in img_names if x.find('.tif') != -1]
        for name in img_names:
            name = name[ : name.find('.tif')]
            self.img_list.append(name)

            if os.path.exists('%s%s.npy' % (self.contour_path , name)):
                self.contour_list.append(np.load('%s%s.npy' % (self.contour_path , name)))
            else:
                self.contour_list.append(np.array([]))

        self.index = 0
        for contour in self.contour_list:
            if len(contour) > 0:
                self.index += 1
            else:
                break

    def init_gui(self):
        self.windows = tkinter.Tk()
        self.windows.title('image annotation')
        self.windows.geometry('%dx%d+100+50' % (
        windows_width, windows_height))  # set width:500 , height:300 , left_margin:100 , up_margin = 50

        self.canvas = tkinter.Canvas(self.windows, width=canvas_width, height=canvas_height)
        self.canvas.place(x=canvas_position[0], y=canvas_position[1])

        self.canvas.bind('<Button-1>', self.canvas_button_1)

        self.button_pred = tkinter.Button(self.windows, width=button_pred_width, height=button_pred_height,
                                     text='pred', command=self.button_pred_clicked)
        self.button_pred.place(x=button_pred_position[0], y=button_pred_position[1])

        self.button_next = tkinter.Button(self.windows, width=button_next_width, height=button_next_height,
                                     text='next', command=self.button_next_clicked)
        self.button_next.place(x=button_next_position[0], y=button_next_position[1])

        self.button_withdraw = tkinter.Button(self.windows, width=button_withdraw_width, height=button_withdraw_height,
                                         text='withdraw', command=self.button_withdraw_clicked)
        self.button_withdraw.place(x=button_withdraw_position[0], y=button_withdraw_position[1])


    def init_image(self , image_index):
        if self.index >= 0 and self.index < len(self.img_list):
            img = Image.open('%s/%s.tif' % (self.sample_path , self.img_list[image_index]))
            contour = self.contour_list[image_index]
            img = img.crop((0 , 0 , 512 , 256))
            img = img.resize((1024 , 512), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(image=img)

            self.canvas.delete('line')
            del_all = lambda x = tkinter.ALL : self.canvas.delete(x)
            del_all() # delete all sub lines

            self.canvas.create_image(0, 0, anchor=tkinter.NW, image=self.img)

            if len(contour) > 0:
                x0, y0 = int(contour[0, 0] * self.ratios), int(contour[0, 1] * self.ratios)
                self.canvas.create_oval(x0 - draw_radius, y0 - draw_radius, x0 + draw_radius,
                                        y0 + draw_radius, fill='red')

            for i in range(1 , len(contour)):
                x0 , y0 = int(contour[i-1 , 0] * self.ratios) , int(contour[i-1 , 1] * self.ratios)
                x1, y1 = int(contour[i, 0] * self.ratios), int(contour[i, 1] * self.ratios)
                self.canvas.create_line(x0 , y0 , x1 , y1 , fill = 'red')
                self.canvas.create_oval(x1 - draw_radius , y1 - draw_radius , x1 + draw_radius ,
                                            y1 + draw_radius, fill = 'red')



    def canvas_button_1(self , event):
        if self.is_work:
            x = event.x
            y = event.y
            if x > 0 and x < 512 and y > 0 and y < 512:
                x , y = int(x // self.ratios) , int(y // self.ratios)
                c_l = list(self.contour_list[self.index])
                c_l.append([x , y])
                self.contour_list[self.index] = np.array(c_l)
                self.init_image(self.index)


    def button_pred_clicked(self):
        if self.is_work:

            if self.index == 0:
                self.index = len(self.img_list) - 1
            else:
                self.index -= 1
            self.init_image(self.index)


    def button_next_clicked(self):
        if self.is_work:
            np.save('%s%s.npy' % (self.contour_path , self.img_list[self.index]) , self.contour_list[self.index])
            self.index = (self.index + 1) % len(self.img_list)
            self.init_image(self.index)

    def button_withdraw_clicked(self):
        if self.is_work:
            c_l = list(self.contour_list[self.index])
            if len(c_l) > 0:
                c_l.pop()
            self.contour_list[self.index] = np.array(c_l)
            self.init_image(self.index)

    def mainloop(self):
        self.windows.mainloop()

if __name__ == '__main__':

    img_path = 'X:/GXB/SNS/data1/Shengfuyou_1th/Positive'

    windows = MainWindow(img_path)
    windows.mainloop()