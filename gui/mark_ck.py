import tkinter
from PIL import Image , ImageTk
import os
from tkinter.filedialog import askdirectory
# from multiprocessing import dummy
# import numpy as np
from gui.tools import *
# from multiprocessing import cpu_count


class MainWindow():

    def __init__(self):

        self.init_gui()
        self.is_work = False
        self.finished_falg = False

        self.work_model = True # 0 represent points , 1 lines
        self.button_points.config(state = tkinter.DISABLED)

        self.new_line_edit = False
        self.new_line_list = []

        self.kernel_contours_state = None
        self.kernel_contours = None

    def init_gui(self):
        self.windows = tkinter.Tk()
        self.windows.title('image annotation')
        # set width:500 , height:300 , left_margin:100 , up_margin = 50
        self.windows.geometry('%dx%d+100+50' % (windows_width, windows_height))

        self.canvas = tkinter.Canvas(self.windows, width=canvas_width, height=canvas_height)
        self.canvas.place(x=canvas_position[0], y=canvas_position[1])

        self.canvas.bind('<MouseWheel>', self.canvas_mouse_wheel)
        self.canvas.bind('<Button-1>', self.canvas_button_1)
        self.canvas.bind('<Button-3>' , self.canvas_button_3)
        self.canvas.bind('<Double-Button-1>' , self.canvas_double_button_1)

        self.list_unfinished = tkinter.Listbox(self.windows, width=list_unfinished_width, height=list_unfinished_height)
        self.list_unfinished.place(x=list_unfinished_position[0], y=list_unfinished_position[1])
        self.list_unfinished.bind('<Double-Button-1>', self.list_unfinished_click)

        self.list_finished = tkinter.Listbox(self.windows, width=list_finished_width, height=list_finished_height)
        self.list_finished.place(x=list_finished_position[0], y=list_finished_position[1])
        self.list_finished.bind('<Double-Button-1>', self.list_finished_click)

        self.button_lines = tkinter.Button(self.windows , width = button_lines_width , height = button_lines_height ,
                                           text = 'lines' , command = self.button_lines_clicked)
        self.button_lines.place(x = button_lines_position[0] , y = button_lines_position[1])

        self.button_del_line = tkinter.Button(self.windows , width = button_del_line_width , height = button_del_line_height ,
                                              text = 'delete' , command = self.button_del_line_clicked)
        self.button_del_line.place(x = button_del_line_positon[0] , y = button_del_line_positon[1])


        self.button_points =tkinter.Button(self.windows , width = button_points_width , height = button_points_height ,
                                           text = 'points' , command = self.button_points_clicked)
        self.button_points.place(x = button_points_position[0] , y = button_points_position[1])

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

            # read result of kernel seg , set state
            if self.kernel_contours_state is None and self.kernel_contours is None:
                self.kernel_contours = list_read('%s/%s.txt' % (self.k_contour_path , self.unfinished_names[self.current_image_index]))
                self.kernel_contours_state = [0] * len(self.kernel_contours)

            img = Image.open(self.img_path + '/' + self.unfinished_names[self.current_image_index] + '.tif')
            self.current_img_original = img.copy()
            self.ori_img_size = img.size
            self.scale_ratio = canvas_width / np.max(self.ori_img_size)
            self.update_image()

    def draw_canvas(self):
        if not self.is_work:
            return

        r = int(d_r_p * self.scale_ratio)
        for point in self.unfinished_points[self.current_image_index]:
            self.draw_point(point , r)

        # draw contours of lesion
        name = self.unfinished_names[self.current_image_index]
        lesion_contours = np.load('%s/%s.npy' % (self.contour_path , name))
        self.draw_contours(lesion_contours)

        # draw new edit line
        r = int(d_r_l * self.scale_ratio)
        if self.new_line_edit:
            if len(self.new_line_list) > 0:
                self.draw_point(self.new_line_list[0] , r)
            for i in range(1 , len(self.new_line_list)):
                self.draw_point(self.new_line_list[i] , r)
                self.draw_line(self.new_line_list[i - 1] , self.new_line_list[i])

        # draw kernel contours
        for k , contour in enumerate(self.kernel_contours):
            self.draw_contours(contour)
            if self.kernel_contours_state[k] == 1:
                for p in contour:
                    self.draw_point(p , r)

    def draw_point(self , p , r):
        p = p.copy()
        p[0], p[1] = self.img_to_canvas(p[0], p[1])
        self.canvas.create_oval(p[0] - r, p[1] - r, p[0] + r, p[1] + r, fill='red')

    def draw_line(self , p0 , p1):
        p0 , p1 = p0.copy() , p1.copy()
        p0[0] , p0[1] = self.img_to_canvas(p0[0], p0[1])
        p1[0] , p1[1] = self.img_to_canvas(p1[0] , p1[1])
        self.canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill='red')

    def draw_contours(self , contour):
        for i in range(len(contour)):
            p1 = contour[i]
            if i == 0:
                p0 = contour[len(contour) - 1]
            else:
                p0 = contour[i - 1]
            self.draw_line(p0 , p1)

    def update_image(self):
        if self.is_work:
            scale_size = (int(self.ori_img_size[0] * self.scale_ratio), int(self.ori_img_size[1] * self.scale_ratio))
            self.scale_size = scale_size
            scale_position = (int((canvas_width - scale_size[0]) / 2), int((canvas_height - scale_size[1]) / 2))
            self.scale_position = scale_position
            img = self.current_img_original.copy().resize(scale_size, Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(image=img)

            # delete all sub points
            del_all = lambda x = tkinter.ALL : self.canvas.delete(x)
            del_all()

            # create img
            self.canvas.create_image(scale_position[0] , scale_position[1] ,  anchor=tkinter.NW, image=self.img)
            # draw mark of img
            self.draw_canvas()

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


    def canvas_to_img(self , x , y):
        if x >= 5 and x <= canvas_width - 5 and y >= 5 and y <= canvas_height - 5:
            x_offset = x - self.scale_position[0]
            y_offset = y - self.scale_position[1]
            if x_offset > 0 and x_offset < self.scale_size[0] and y_offset > 0 and y_offset < self.scale_size[1]:
                x = int(x_offset / self.scale_ratio)
                y = int(y_offset / self.scale_ratio)
                return x , y
        return -1 , -1

    def img_to_canvas(self , x , y):
        x = int(x * self.scale_ratio) + self.scale_position[0]
        y = int(y * self.scale_ratio) + self.scale_position[1]
        return x , y

    def canvas_button_1(self , event):
        print('canvas button_1 clicked ... ')
        if self.is_work: # click
            x , y = self.canvas_to_img(event.x , event.y)
            if x != -1 and y != -1:
                if self.work_model:
                    self.unfinished_points[self.current_image_index].append([x , y])
                elif not self.work_model:
                    self.kernel_contours_state = [0] * len(self.kernel_contours)
                    self.new_line_edit = True
                    self.new_line_list.append([x , y])
                self.update_image()

    def canvas_button_3(self , event):
        print('canvas button_3 clicked ...')
        if self.is_work and not self.work_model and not self.new_line_edit:
            x , y = self.canvas_to_img(event.x , event.y)
            # judge wether choosen
            dis = max_distance
            k = -1
            for i , contour in enumerate(self.kernel_contours):
                self.kernel_contours_state[i] = 0
                contour = np.array(contour)
                l = np.min(contour[: , 0])
                r = np.max(contour[: , 0])
                t = np.min(contour[: , 1])
                d = np.max(contour[: , 1])
                if x > l and x < r and y > t and y < d:
                    cx = (l + r) / 2
                    cy = (t + d) / 2
                    c_dis = np.square((x - cx) ** 2 + (y - cy) ** 2)
                    if c_dis < dis:
                        dis = c_dis
                        k = i
            if k != -1:
                self.kernel_contours_state[k] = 1

            self.update_image()



    def canvas_double_button_1(self , event):
        print('canvas_double_button_1 clicked ...')
        if self.is_work and not self.work_model:
            self.new_line_edit = False
            x, y = event.x, event.y
            if len(self.new_line_list) > 2:
                self.kernel_contours.append(self.new_line_list)
                self.kernel_contours_state.append(0)
                self.new_line_list = []
            self.update_image()

    def list_unfinished_click(self , event):
        if self.is_work:
            index = self.list_unfinished.curselection()
            self.kernel_contours_state = None
            self.kernel_contours = None
            self.init_image(index[0])

    def list_finished_click(self , event):
        if self.is_work:
            print(self.list_finished.curselection())

    def button_finished_clicked(self):
        if self.is_work: # end edit current img

            # save points
            np.save(self.contour_path + '/p_' + self.unfinished_names[self.current_image_index] ,
                    np.array(self.unfinished_points[self.current_image_index]))

            # save contours
            list_save('%s/c_%s.txt' % (self.k_contour_path , self.unfinished_names[self.current_image_index]) ,
                      self.kernel_contours)

            self.finished_points.append(self.unfinished_points[self.current_image_index])
            self.finished_names.append(self.unfinished_names[self.current_image_index])
            self.unfinished_names.remove(self.unfinished_names[self.current_image_index])
            self.unfinished_points.remove(self.unfinished_points[self.current_image_index])

            self.list_unfinished.delete(self.current_image_index)
            self.list_finished.insert(tkinter.END , self.finished_names[len(self.finished_names) - 1])

            if len(self.unfinished_names) <= 0:
                self.is_work = False

            try:
                self.kernel_contours = None
                self.kernel_contours_state = None
                self.init_image((self.current_image_index) % len(self.unfinished_names))
            except:
                print('finished image annotation ... ')

    def button_points_clicked(self):
        self.work_model = not self.work_model
        self.button_lines.config(state = tkinter.ACTIVE)
        self.button_points.config(state = tkinter.DISABLED)

    def button_lines_clicked(self):
        self.work_model = not self.work_model
        self.button_lines.config(state = tkinter.DISABLED)
        self.button_points.config(state = tkinter.ACTIVE)

    def button_next_clicked(self):
        if self.is_work:
            index = (self.current_image_index + 1) % len(self.unfinished_names)
            self.kernel_contours = None
            self.kernel_contours_state = None
            self.init_image(index)

    def button_withdraw_clicked(self):
        if self.is_work and self.work_model:
            try:
                self.unfinished_points[self.current_image_index].pop()
                self.update_image()
            except:
                print('withdraw failed ... ')

    def button_del_line_clicked(self):
        if self.is_work and not self.work_model:
            print('del clicked ...')
            k = -1
            for i , f in enumerate(self.kernel_contours_state):
                if f == 1:
                    k = i
            if k != -1:
                self.kernel_contours_state.pop(k)
                self.kernel_contours.pop(k)
                self.update_image()


    def button_choose_file_clicked(self):

        path = askdirectory()
        self.path = path
        self.label_path.delete(0.0, tkinter.END)
        self.label_path.insert(tkinter.END , path)

        self.c_img_path = path + '/C_Img'
        self.img_path = path + '/Img'
        self.contour_path = path + '/Contours'
        self.k_contour_path = path + '/K_Contours'

        self.img_names = os.listdir(self.img_path)
        self.img_names = [x for x in self.img_names if x.find('.tif') != -1]

        self.is_work = False
        if self.img_names.__len__() > 0:
            self.is_work = True

        if self.is_work:
            self.init_unn_list()
            self.init_image(0)

    def mainloop(self):
        self.windows.mainloop()

if __name__ == '__main__':

    windows = MainWindow()
    windows.mainloop()