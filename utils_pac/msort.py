

class MSort():

    def __init__(self):
        pass

    def swap(self , list , i , j):
        t = list[i]
        list[i] = list[j]
        list[j] = t

    def sift(self , list , i , size):

        while 2 * i + 1 < size:

            left = 2 * i + 1
            right = 2 * i + 2

            if right < size:
                if list[right] <= list[i] and list[left] <= list[i]:
                    return
                else:
                    j = left if list[left] > list[right] else right
                    self.swap(list , i , j)
                    i = j
            else:
                if list[left] > list[i]:
                    self.swap(list , i ,left)
                    i = left
                else:
                    return

    def sift_s(self , list, names, i, size):
        while 2 * i + 1 < size:

            left = 2 * i + 1
            right = 2 * i + 2

            if right < size:
                if list[right] <= list[i] and list[left] <= list[i]:
                    return
                else:
                    j = left if list[left] > list[right] else right
                    self.swap(list , i , j)
                    self.swap(names , i , j)
                    i = j
            else:
                if list[left] > list[i]:
                    self.swap(list , i ,left)
                    self.swap(names , i , left)
                    i = left
                else:
                    return

    def heap_sort(self , list):

#         init heap
        size = len(list)
        for i in range(int(size // 2) , -1 , -1):
            self.sift(list , i , size)

#         order construct
        for i in range(size - 1 , 0 , -1):
            self.swap(list , 0 , i)
            self.sift(list , 0 , i)

        return list


#     协同排序
    def heap_sort_s(self , list , names):

        assert len(list) == len(names)

        size = len(list)
        for i in range(int(size // 2) , -1 , -1):
            self.sift_s(list , names , i , size)

        for i in range(size - 1 , 0 , -1):
            self.swap(list , 0 , i)
            self.swap(names , 0 , i)
            self.sift_s(list , names , 0 , i)

        return list , names

if __name__ == '__main__':

    list = [1 , 2  , 0 , 1 , 0.9]
    names = ['1' , '2' , '0' , '1' , '0.9']
    list , names = MSort().heap_sort_s(list , names)
    print(list , names)
    pass
