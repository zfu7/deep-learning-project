import csv

KEY_LABEL = 4
KEY_TEXT = 1

label = []
text = []

data_path = '../data/uci-news-aggregator.csv'

with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')

    next(reader, None)

    for index, row in enumerate(reader):
        label.append(row[KEY_LABEL])
        text.append(row[KEY_TEXT].lower()[:row[KEY_TEXT].find("https://")-1])

print(len(label), len(text))
print(text[1])
