
import jieba

text_seg_all_train=[]
with open("D:\python project me\data\my_point_net/train/train_content.txt","r",encoding="utf-8") as f1:
    for line in f1.readlines():
        seg = jieba.cut(line)
        text_seg=' '.join(seg)
        text_seg_all_train.append(text_seg)
with open("D:\python project me\data\my_point_net/train/train.txt","w") as f2:
    for i in text_seg_all_train:
       f2.write(i)

text_seg_all_val=[]
with open("D:\python project me\data\my_point_net/val/val_content.txt","r",encoding="utf-8") as f1:
    for line in f1.readlines():
        seg = jieba.cut(line)
        text_seg=' '.join(seg)
        text_seg_all_val.append(text_seg)
with open("D:\python project me\data\my_point_net/val/val.txt","w") as f2:
    for i in text_seg_all_val:
       f2.write(i)

text_seg_all_test=[]
with open("D:\python project me\data\my_point_net/test/test_content.txt","r",encoding="utf-8") as f1:
    for line in f1.readlines():
        seg = jieba.cut(line)
        text_seg=' '.join(seg)
        text_seg_all_test.append(text_seg)
with open("D:\python project me\data\my_point_net/test/test.txt","w") as f2:
    for i in text_seg_all_test:
       f2.write(i)