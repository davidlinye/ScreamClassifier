import csv

filename = "resources/dataset/Annotations/final/annotations_api.csv"

classes = ["clean", "fry", "layered"]
clean_time = fry_time = layered_time = 0


def check_classes(input):
    global clean_time
    global fry_time
    global layered_time

    component_length = (float(row[4]) - float(row[3]))
    if (input == classes[0]):
        clean_time += component_length
    elif (input == classes[1]):
        fry_time += component_length
    elif (input == classes[2]):
        layered_time += component_length
    else:
        print("UNCLASSIFIED", row[5])
        return -1


with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            if check_classes(row[5]) == -1:
                print(row)

            line_count += 1
    print(f'Processed {line_count} lines.')

print("Second of clean: ", clean_time)
print("Second of fry: ", fry_time)
print("Second of layered: ", layered_time)
print("Total seconds: ", clean_time + fry_time + layered_time)
