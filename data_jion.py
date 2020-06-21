


with open("D:\python project me\data\my_point_net/finished_files\exp_logs\decode\GENERATED_SUMMARY.txt","r") as f1:
    for line in f1.readlines():
        line1 = line.replace(" ", "")
with open("D:\python project me\data\my_point_net/finished_files\exp_logs\decode\GENERATED_SUMMARY_jion.txt","w") as f2:
    f2.write(line1)





