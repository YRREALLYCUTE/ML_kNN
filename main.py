import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import random as rd
import matplotlib.pyplot as plt
from PIL import Image


# 导入数据
def load_file(filename):
    x = open(filename)
    array_of_lines = x.readlines()
    number_of_lines = len(array_of_lines)
    return_mat = np.zeros((number_of_lines, 266))
    class_label_vector = []
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split(' ')
        return_mat[index, :] = list_from_line[0:266]

        class_label_vector.append(int(list_from_line[-1]))
        index += 1

    print(type(return_mat))
    return return_mat


# 返回未排序的结果
def get_distance(data1, data2):
    distance_list = []
    for index in range(data2.shape[0]):
        vec = data2[index, :]
        distance_list.append(euclidean(data1, vec))
    return distance_list


# 计算向量之间的距离
def euclidean(v1, v2):
    sum = 0.0
    for i in range(len(v2)):
        if v1[i] != v2[i]:
            sum += 1
    return sum


# 取距离最近的前k个值，并记录其对应的下标
def knn(distance_list, k):
    temp = np.argsort(distance_list)
    temp_res = []
    for i in range(k + 1):
        temp_res.append(temp[i])
    return temp_res


# 分析得到的k个值，确定包含最多的类别，对于相同的类别，按照距离进行判断
def get_result(data_result, temp_res):
    res = []
    real = -1

    for i in range(len(temp_res)):
        temp = data_result[temp_res[i], :]
        for j in range(len(temp)):
            if temp[j] == 1:
                if i != 0:
                    res.append(j)
                else:
                    real = j
                break
    count = np.bincount(res)
    result = np.argmax(count)
    return result, real


# 原始数据图形的绘制
def draw_pic(train_set, result_set):
    pic = np.array(range(0, 256)).reshape(16, 16)
    res_e = 0
    for j in range(train_set.shape[0]):
        temp_train = train_set[j, :]
        temp_res = result_set[j, :]
        for i in range(len(temp_train)):
            pic[int(i / 16)][i % 16] = 255 - temp_train[i] * 255
        for k in range(len(temp_res)):
            if temp_res[k] == 1:
                res_e = k

        new_im = Image.fromarray(pic)
        path = r'E:\工作学习\大三下\机器学习\project\kNN_hmw\im\origin\_origin_' + str(res_e) + '_' + str(j) + '.jpg'
        new_im = new_im.convert('RGB')
        new_im.save(path)


# 识别结果与原始数据的对比
def pic_rec_ori(raw_index, train_set, res_index, real, res):
    pic = np.array(range(0, 16 * 34)).reshape(16, 34)
    temp_ori = train_set[raw_index, :]
    temp_res = train_set[res_index, :]

    for i in range(len(temp_ori)):
        pic[int(i / 16)][i % 16] = 255 - temp_ori[i] * 255
        pic[int(i / 16)][16] = 255
        pic[int(i / 16)][17] = 255
        pic[int(i / 16)][i % 16 + 18] = 255 - temp_res[i] * 255

    new_im = Image.fromarray(pic)
    path = r'E:\工作学习\大三下\机器学习\project\kNN_hmw\im\recognize\_ori_' \
           + str(real) + '_rec' + str(res) + '_' + str(raw_index) + '.jpg'
    new_im = new_im.convert('RGB')
    new_im.save(path)


# 柱形图绘制函数
def plain_rect(data_num):
    font_size = 10
    fig_size = (8, 6)

    names = (u'Accurate', u'Inaccurate')
    subjects = (u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9')
    scores = ((data_num[0][0], data_num[1][0], data_num[2][0],data_num[3][0],data_num[4][0],
               data_num[5][0],data_num[6][0], data_num[7][0],data_num[8][0], data_num[9][0]),
              (data_num[0][1],data_num[1][1],data_num[2][1],data_num[3][1],data_num[4][1],
               data_num[5][1],data_num[6][1], data_num[7][1], data_num[8][1], data_num[9][1]))

    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['figure.figsize'] = fig_size
    bar_with = 0.30

    index_u = np.arange(len(scores[0]))
    rects1 = plt.bar(index_u, scores[0], bar_with, color='#0072BC', label=names[0])
    rects2 = plt.bar(index_u, scores[1], bar_with, color='#ED1C24', label=names[1])

    plt.xticks(index_u+bar_with,subjects)
    plt.ylim(ymax = 180, ymin = 0)

    plt.title(u'Comparisons of Accurate and Inaccurate Numbers of Each Number')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),fancybox = True, ncol = 5)

    # 添加数据标签
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')
            # 柱形图边缘用白色填充，纯粹为了美观
            rect.set_edgecolor('white')

    add_labels(rects1)
    add_labels(rects2)

    plt.savefig('compare.png')
    plt.show()


if __name__ == '__main__':
    rate = 1  # 通过更改rate的值，可以改变测试集的大小
    N = 1  # 这是k值取值范围，1 - N

    # 初始数据读入
    data = load_file('semeion.data')
    data_train = data[:, 0:255]
    data_result = data[:, 256:266]

    # 绘制数据的图示
    # draw_pic(data_train, data_result)
    # 随机选取一部分作为测试集
    test_set = data_train
    item_to_del = []
    for index_o in range(data_train.shape[0]):
        if rd.random() > rate:
            item_to_del.append(index_o)
    test_set = np.delete(test_set, item_to_del, axis=0)
    print("测试集大小为：", test_set.shape[0])

    # 计算并输出准确率
    results = []
    for k in range(N):
        # 用于统计正确率
        count = 0
        count_num = np.array(range(0, 20)).reshape(10, 2)
        for i in range(10):
            for j in range(2):
                count_num[i][j] = 0  # 初始化矩阵

        for index in range(test_set.shape[0]):
            row = test_set[index, :]
            temp = data_train
            dis_list = get_distance(row, temp)
            k_res = knn(dis_list, k + 1)
            res_end = get_result(data_result, k_res)
            if res_end[0] == res_end[1]:
                count = count + 1
            # print(res_end[0],"-", res_end[1])  # 每一次的预测结果和准确值
            if index % 30 == 0:
                print("======", index, "====")
            # 绘制对比图像
            #if k == 0:
            #    pic_rec_ori(index, data_train, k_res[1], res_end[1], res_end[0])

            # 绘制柱形图的数据统计
            if res_end[0] == res_end[1]:
                count_num[res_end[0]][0] += 1  # 预测准确
            else:
                count_num[res_end[1]][1] += 1  # 预测不准确

        print(count / test_set.shape[0])  # 打印准确率
        # 用于绘制k的取值与准确率关系的图像
        results.append(count / test_set.shape[0])

        # 绘制柱形图
        plain_rect(count_num)

    X = np.linspace(0, N, N, endpoint=True)
    plt.plot(X, results)

    plt.show()
