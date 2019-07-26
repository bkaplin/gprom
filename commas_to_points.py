files = ["../train_data.csv", ]#"../need_fill_data.csv"]
for file_name in files:
    f = open(file_name, 'r')
    data = f.readlines()
    f.close()

    new_file_name = file_name.replace("../", "")
    new_csv = ""
    for line in data:
        correct_line = ""
        for elem in line.split(";"):
            if elem == "0" or elem == "0.0":
                new_elem = ""
            else:
                new_elem = elem.replace(",", ".").replace(" ", "")
            correct_line += new_elem + ";"
        new_csv += correct_line.rstrip(";") + "\n"
    f = open("correct_csv/" + new_file_name[:-4] + "_new" + new_file_name[-4:], "w")
    f.write(new_csv)
    f.close()
