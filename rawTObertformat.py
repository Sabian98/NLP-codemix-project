import csv

#train/dev
# with open('Hinglish/Hinglish_dev_3k_split_conll.tsv', 'wt') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     lines = open('Hinglish/Hinglish_dev_3k_split_conll.txt', encoding='utf-8').read().strip().split('\n')

#     sentence = ""
#     label = ""
#     for line in lines:
#         if line.find("meta\t") != -1:
#             if label == "negative":
#                 tsv_writer.writerow([sentence, 0])
#             if label == "neutral":
#                 tsv_writer.writerow([sentence, 1])
#             if label == "positive":
#                 tsv_writer.writerow([sentence, 2])
#             label = line.split("\t")[2]
#             sentence = ""
#         else:
#             sentence += line.split("\t")[0]
#             sentence += " "

#test
# with open('Hinglish/Hinglish_test_labeled_conll_updated.tsv', 'wt') as out_file:
#     tsv_writer = csv.writer(out_file, delimiter='\t')
#     lines1 = open('Hinglish/Hinglish_test_unlabeled_conll_updated.txt', encoding='utf-8').read().strip().split('\n')
#     lines2 = open('Hinglish/Hinglish_test_labels.txt', encoding='utf-8').read().strip().split('\n')

#     sentence = ""
#     label = 0
#     for line in lines1:
#         if line.find("meta\t") != -1:
#             if lines2[label].split(",")[1] == "negative":
#                 tsv_writer.writerow([sentence, 0])
#             if lines2[label].split(",")[1] == "neutral":
#                 tsv_writer.writerow([sentence, 1])
#             if lines2[label].split(",")[1] == "positive":
#                 tsv_writer.writerow([sentence, 2])
#             label += 1
#             sentence = ""
#         else:
#             sentence += line.split("\t")[0]
#             sentence += " "