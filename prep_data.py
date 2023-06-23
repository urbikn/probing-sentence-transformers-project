import numpy as np


def find_label(task, line):

    if task == 'main':
        line[1] = line[1].split()[0]
    elif task == 'fourth':
        line[1] = line[0].split()[3]
    elif task == 'second':
        line[1] = line[0].split()[1]

    return line


def filter_decl(lines):

    filtered_lines = []
    for line in lines:
        item = line.split('\t')[0]
        target = line.split('\t')[1]
        if item.strip().endswith('quest'):
            item_stripped = ' '.join(item.split()[:-1])
            filtered_lines.append([item_stripped, target])

    return filtered_lines


def data_splits(percentages, lines):

    train, validate, test = np.split(lines, [int(len(lines)*percentages[0]), int(len(lines)*(percentages[0] + percentages[1]))])

    return (train, test, validate)


def generate_unified_format(file, task, percentages, filename):

    with open(file, 'r') as f:
        lines = f.readlines()

    filtered_lines = filter_decl(lines)

    final_line_contents = []
    for i, line in enumerate(filtered_lines):
        final_line_contents.append(find_label(task, line))

    ttv_splits = data_splits(percentages, final_line_contents)

    with open(filename, 'w') as fn:

        for i, split in enumerate(ttv_splits):
            for line in split:
                if i == 0:
                    fn.write('tr\t{0}\t{1}\n'.format(line[1], line[0]))
                elif i == 1:
                    fn.write('te\t{0}\t{1}\n'.format(line[1], line[0]))
                else:
                    fn.write('va\t{0}\t{1}\n'.format(line[1], line[0]))
    

    
if __name__ == "__main__":

    generate_unified_format('data/question.all', 'fourth', [(10/12), (1/12)], 'pov_questions_fourth.txt')
