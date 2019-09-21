import numpy as np

'''
tools function
'''

def list_save(path , list_data):
    with open(path , 'w') as file:
        for d in list_data:
            if len(np.shape(d)) > 2:
                d = d[: , 0  ,:]
            for item in d:
                file.write('%d_%d-' % (item[0] , item[1]))
            file.write('\n')

def list_read(path):
    with open(path , 'r') as file:
        d_list = []
        for line in file:
            line = line[ : -1]
            line = line.split('-')
            d = []
            for item in line:
                uints = item.split('_')
                if len(uints) > 1:
                    d.append([int(uints[0]) , int(uints[1])])
            d_list.append(d)
        return d_list

'''
 windows paramter setting
'''

windows_width = 1100
windows_height = 850

canvas_width = 800
canvas_height = 700
canvas_position = (150 , 50)

list_unfinished_width = 18
list_unfinished_height = 39
list_unfinished_position = (10 , 50)

list_finished_width = 18
list_finished_height = 39
list_finished_position = (960 , 50)

button_lines_width = 20
button_lines_height = 1
button_lines_position = (150 , 800)

button_del_line_width = 20
button_del_line_height = 1
button_del_line_positon = (350 , 800)

button_points_width = 20
button_points_height = 1
button_points_position = (150 , 760)

button_finished_width = 20
button_finished_height = 1
button_finished_position = (350 , 760)

button_next_width = 20
button_next_height = 1
button_next_position = (550 , 760)

button_withdraw_width = 20
button_withdraw_height = 1
button_withdraw_position = (750 , 760)

button_choose_file_width = 20
button_choose_file_height = 1
button_choose_file_position = (10 , 10)

label_path_width = 100
label_path_height = 1
label_path_position = (180 , 10)

text_unfinished_width = 20
text_unfinished_height = 1
text_unfinished_position = (10 , 760)

text_finished_width = 18
text_finished_height = 1
text_finished_position = (960 , 760)

d_r_p = 4
d_r_l = 2
max_distance = 10000000