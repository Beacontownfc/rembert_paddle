
dataset_dir = ['de/test_2k.tsv', 'en/test_2k.tsv',
               'es/test_2k.tsv', 'fr/test_2k.tsv',
               'ja/test_2k.tsv', 'ko/test_2k.tsv',
               'zh/test_2k.tsv'
               ]
wf = open('test_2k.tsv', 'w', encoding='utf-8', newline='')

for i, dataset in enumerate(dataset_dir):
    with open(dataset, 'r', encoding='utf-8') as rf:
        for j, line in enumerate(rf):
            if i == 0 and j == 0:
                wf.write(line)
            elif j == 0 and i != 0:
                continue
            else:
                wf.write(line)

wf.close()
