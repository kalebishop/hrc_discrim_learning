import csv

COLOR_I = 0
SIZE_I = 1
DIM_I = 2

COLOR = ["red", "yellow", "blue", "green", "purple", "grey", "white"]
SIZE = ["big", "biggest", "small", "smallest"]
DIM = ["long", "longest", "loing", "short", "shortest", "length"]

def get_usage_counts_per_stim():
    # read from csv titled latest.csv
    with open("data/latest.csv", newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        header = next(csvreader)
        stim_dict = {}

        qs_to_indicies = {}
        indicies_to_qs = {}

        # get field names from first row
        for i in range(len(header)):
            field = header[i]
            # print(field)
            if field[0] == "Q":
                indicies_to_qs[i] = field
                qs_to_indicies[field] = i

                stim_dict[field] = [0, 0, 0]

        row = next(csvreader)
        # iterate over first 40 samples for testing
        for row in csvreader:
            for index in indicies_to_qs.keys():
                response = row[index]
                # print(response)

                # analyze response feature usage
                entry = stim_dict[indicies_to_qs[index]]

                for c in COLOR:
                    if c in response:
                        entry[COLOR_I] += 1
                        break
                for s in SIZE:
                    if s in response:
                        entry[SIZE_I] += 1
                        break
                for d in DIM:
                    if d in response:
                        entry[DIM_I] += 1
                        break

    return stim_dict

def write_csv_from_dict(dict, output_filename, header_fields=[]):
    with open(output_filename, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        csvwriter.writerow(header_fields)
        for key in dict:
            csvwriter.writerow([key] + dict[key])

def get_usage_counts_per_pid():
    raise NotImplementedError

def main():
    # pass
    dict = get_usage_counts_per_stim()
    # print(dict)
    header_fields = ["qid", "color", "size", "dim"]
    output_filename = "data/data_by_qid.csv"

    write_csv_from_dict(dict, output_filename, header_fields)

if __name__ == "__main__":
    main()
