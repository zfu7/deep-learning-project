import csv

data_path = '../data/tweets.csv'

label = []
data = []

KEY_LABEL = 1
KEY_TEXT = 2

lowercase = True

with open(data_path, 'r') as f:
    csv_reader = csv.reader(f, delimiter=',', quotechar='"')

    next(csv_reader, None)

    for index, row in enumerate(csv_reader):

        if row[KEY_LABEL] == "realDonaldTrump":
            label.append(0)
        if row[KEY_LABEL] == "HillaryClinton":
            label.append(1)

        data.append(row[KEY_TEXT].lower()[:row[KEY_TEXT].find("https://")-1])

print(len(data))
print(len(label))

# for d in data:
#     for idx, c in enumerate(d):
#         print(idx, c)
#     break