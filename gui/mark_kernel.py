import tkinter
from PIL import Image , ImageTk
import os
from tkinter.filedialog import askdirectory
from multiprocessing import dummy
import numpy as np
# from multiprocessing import cpu_count

windows_width = 900
windows_height = 700

canvas_width = 600
canvas_height = 600
canvas_position = (150 , 50)

list_unfinished_width = 18
list_unfinished_height = 34
list_unfinished_position = (10 , 50)

list_finished_width = 18
list_finished_height = 34
list_finished_position = (760 , 50)

button_finished_width = 20
button_finished_height = 1
button_finished_position = (150 , 660)

button_next_width = 20
button_next_height = 1
button_next_position = (350 , 660)

button_withdraw_width = 20
button_withdraw_height = 1
button_withdraw_position = (550 , 660)

button_choose_file_width = 20
button_choose_file_height = 1
button_choose_file_position = (10 , 10)

label_path_width = 80
label_path_height = 1
label_path_position = (180 , 10)

text_unfinished_width = 20
text_unfinished_height = 1
text_unfinished_position = (10 , 670)

text_finished_width = 18
text_finished_height = 1
text_finished_position = (740 , 670)

draw_radius = 4

class MainWindow():

    def __init__(self):

        self.init_gui()
        self.is_work = False
        self.finished_falg = False

    def init_gui(self):
        self.windows = tkinter.Tk()
        self.windows.title('image annotation')
        self.windows.geometry('%dx%d+100+50' % (
        windows_width, windows_height))  # set width:500 , height:300 , left_margin:100 , up_margin = 50

        self.canvas = tkinter.Canvas(self.windows, width=canvas_width, height=canvas_height)
        self.canvas.place(x=canvas_position[0], y=canvas_position[1])

        self.canvas.bind('<MouseWheel>', self.canvas_mouse_wheel)
        self.canvas.bind('<Button-1>', self.canvas_button_1)
        self.canvas.bind('<B1-Motion>', self.canvas_b1_motion)
        self.canvas.bind('<Motion>', self.canvas_motion)

        self.list_unfinished = tkinter.Listbox(self.windows, width=list_unfinished_width, height=list_unfinished_height)
        # for i in range(0, 100):
        #     self.list_unfinished.insert(i, str(i))
        self.list_unfinished.place(x=list_unfinished_position[0], y=list_unfinished_position[1])
        self.list_unfinished.bind('<Double-Button-1>', self.list_unfinished_click)

        self.list_finished = tkinter.Listbox(self.windows, width=list_finished_width, height=list_finished_height)
        # for i in range(0, 100):
        #     self.list_finished.insert(i, str(i))
        self.list_finished.place(x=list_finished_position[0], y=list_finished_position[1])
        self.list_finished.bind('<Double-Button-1>', self.list_finished_click)

        self.button_finished = tkinter.Button(self.windows, width=button_finished_width, height=button_finished_height,
                                         text='finished', command=self.button_finished_clicked)
        self.button_finished.place(x=button_finished_position[0], y=button_finished_position[1])

        self.button_next = tkinter.Button(self.windows, width=button_next_width, height=button_next_height,
                                     text='next', command=self.button_next_clicked)
        self.button_next.place(x=button_next_position[0], y=button_next_position[1])

        self.button_withdraw = tkinter.Button(self.windows, width=button_withdraw_width, height=button_withdraw_height,
                                         text='withdraw', command=self.button_withdraw_clicked)
        self.button_withdraw.place(x=button_withdraw_position[0], y=button_withdraw_position[1])

        self.label_path = tkinter.Text(self.windows, width=label_path_width, height=label_path_height)
        self.label_path.place(x=label_path_position[0], y=label_path_position[1])

        self.button_choose_file = tkinter.Button(self.windows, width=button_choose_file_width, height=button_choose_file_height,
                                            text='choose_file', command=self.button_choose_file_clicked)
        self.button_choose_file.place(x=button_choose_file_position[0], y=button_choose_file_position[1])


        self.text_unfinished = tkinter.Text(self.windows , width = text_finished_width , height = text_finished_height)
        self.text_unfinished.place(x = text_unfinished_position[0] , y = text_unfinished_position[1])

        self.text_finished = tkinter.Text(self.windows , width = text_finished_width , height = text_finished_height)
        self.text_finished.place(x = text_finished_position[0] , y = text_finished_position[1])

    def init_image(self , image_index):
        if self.is_work:
            self.current_image_index = image_index
            self.label_path.delete(0.0, tkinter.END)
            self.label_path.insert(tkinter.END, self.path + '/' + self.unfinished_names[self.current_image_index] + '.tif')

            self.text_finished.delete(0.0 , tkinter.END)
            self.text_finished.insert(tkinter.END , 'f nums : ' + str(len(self.finished_names)))

            self.text_unfinished.delete(0.0 , tkinter.END)
            self.text_unfinished.insert(tkinter.END , 'uf nums : ' + str(len(self.unfinished_names)))

            img = Image.open(self.c_img_path + '/' + self.unfinished_names[self.current_image_index] + '.tif')
            self.current_img_original = img.copy()
            self.ori_img_size = img.size
            self.scale_ratio = canvas_width / np.max(self.ori_img_size)
            self.update_image()

    def draw_points(self):
        if self.is_work:
            print('draw points' , self.current_image_index , len(self.unfinished_points[self.current_image_index]))
            for point in self.unfinished_points[self.current_image_index]:
                p = point.copy()
                print(p , self.ori_img_size)
                p[0] = int(p[0] * self.scale_ratio) + self.scale_position[0]
                p[1] = int(p[1] * self.scale_ratio) + self.scale_position[1]
                r = int(draw_radius * self.scale_ratio)
                self.canvas.create_oval(p[0] - r , p[1] - r ,
                                        p[0] + r , p[1] + r, fill='red')

    def update_image(self):
        if self.is_work:
            scale_size = (int(self.ori_img_size[0] * self.scale_ratio), int(self.ori_img_size[1] * self.scale_ratio))
            self.scale_size = scale_size
            scale_position = (int((canvas_width - scale_size[0]) / 2), int((canvas_height - scale_size[1]) / 2))
            self.scale_position = scale_position
            img = self.current_img_original.copy().resize(scale_size, Image.ANTIALIAS)
            self.t_img = ImageTk.PhotoImage(image=img)
            self.canvas.delete('oval')
            del_all = lambda x = tkinter.ALL : self.canvas.delete(x)
            del_all() # delete all sub points
            self.canvas.create_image(scale_position[0] , scale_position[1] ,  anchor=tkinter.NW, image=self.t_img)
            self.draw_points()

    def init_unn_list(self):
        if self.is_work:
            self.unfinished_names = []
            self.unfinished_points = []
            self.finished_names = []
            self.finished_points = []
            for i in range(0 , len(self.img_names)):
                name = self.img_names[i][ : self.img_names[i].find('.tif')]
                if os.path.exists(self.contour_path + '/p_' + name + '.npy'):
                    self.finished_names.append(name)
                    points = np.load(self.contour_path + '/p_' + name + '.npy')
                    self.finished_points.append(points)
                    self.list_finished.insert(tkinter.END , name)
                else:
                    self.unfinished_names.append(name)
                    self.unfinished_points.append([])
                    self.list_unfinished.insert(tkinter.END , name)

    def canvas_mouse_wheel(self , event):
        if self.is_work:
            if event.delta > 0: #up
                if self.scale_ratio <= 2:
                    self.scale_ratio += 0.2
            else: # down
                if self.scale_ratio >= 0.5:
                    self.scale_ratio -= 0.2
            self.update_image()

    def canvas_button_1(self , event):
        if self.is_work: # double click
            self.pre_x , self.pre_y = event.x , event.y
            if self.pre_x >= 5 and self.pre_x <= canvas_width - 5 and\
                self.pre_y >= 5 and self.pre_y <= canvas_height - 5:

                x_offset = self.pre_x - self.scale_position[0]
                y_offset = self.pre_y - self.scale_position[1]
                print(x_offset , y_offset)

                if x_offset > 0 and x_offset < self.scale_size[0] and y_offset > 0 and y_offset < self.scale_size[1]:
                    x = int(x_offset / self.scale_ratio)
                    y = int(y_offset / self.scale_ratio)
                    print(x , y)
                    self.unfinished_points[self.current_image_index].append([x , y])
                    self.update_image()

    def canvas_b1_motion(self , event):
        if self.is_work:
            print('b1_motion' , event.x , event.y)


    def canvas_motion(self , event):
        if self.is_work:
            pass
            # print(event.x , event.y)

    def list_unfinished_click(self , event):
        if self.is_work:
            index = self.list_unfinished.curselection()
            self.init_image(index[0])

    def list_finished_click(self , event):
        if self.is_work:

            print(self.list_finished.curselection())

    def button_finished_clicked(self):
        if self.is_work:

            np.save(self.contour_path + '/p_' + self.unfinished_names[self.current_image_index] ,
                    np.array(self.unfinished_points[self.current_image_index]))

            self.finished_points.append(self.unfinished_points[self.current_image_index])
            self.finished_names.append(self.unfinished_names[self.current_image_index])
            self.unfinished_names.remove(self.unfinished_names[self.current_image_index])
            self.unfinished_points.remove(self.unfinished_points[self.current_image_index])

            self.list_unfinished.delete(self.current_image_index)
            self.list_finished.insert(tkinter.END , self.finished_names[len(self.finished_names) - 1])

            if len(self.unfinished_names) <= 0:
                self.is_work = False

            try:
                self.init_image((self.current_image_index) % len(self.unfinished_names))
            except:
                print('finished image annotation ... ')

    def button_next_clicked(self):
        if self.is_work:
            index = (self.current_image_index + 1) % len(self.unfinished_names)
            self.init_image(index)

    def button_withdraw_clicked(self):
        if self.is_work:
            try:
                self.unfinished_points[self.current_image_index].pop()
                self.init_image(self.current_image_index)
            except:
                print('withdraw failed ... ')
            # print('button_widthdraw_clicked..')

    def button_choose_file_clicked(self):
        # print('button_choose_file_clicked...')
        path = askdirectory()
        self.path = path
        self.label_path.delete(0.0, tkinter.END)
        self.label_path.insert(tkinter.END , path)

        self.c_img_path = path + '/C_Img'
        self.img_path = path + '/Img'
        self.contour_path = path + '/Contours'

        # print(self.c_img_path , self.img_path , self.contour_path)
        self.img_names = os.listdir(self.img_path)
        self.img_names = [x for x in self.img_names if x.find('.tif') != -1]

        self.is_work = False
        if self.img_names.__len__() > 0:
            self.is_work = True

        print(self.is_work)
        if self.is_work:
            self.init_unn_list()
            self.init_image(0)

    def mainloop(self):
        self.windows.mainloop()

if __name__ == '__main__':

    windows = MainWindow()
    windows.mainloop()