import os
directory_list = []
already_todo = False
with open('output.md', 'wb') as doc:
    for file_ in os.listdir(os.getcwd()):
        if file_.endswith('.py') and file_ != 'make_doc.py':
            with open(file_, 'r') as code:
                doc.write('#{!s}\n'.format(file_))
                text = code.read()
                lines = text.split('\n')
                for line in lines:
                    line = line.lstrip()
                    if line[0:6] == 'import':
                        doc.write('* {!s}\n'.format(line))
                for line in lines:
                    line = line.lstrip().replace('_', '\_')
                    if line[0:3] == 'def':
                        doc.write('## {!s}\n'.format(line.replace('def ', '')))
                    if line[0:1] == '#':
                        if line[0:7] == '# TODO:':
                            if already_todo is False:
                                doc.write('## TO DO:\n')
                                already_todo = True
                            doc.write('*{!s}\n'.format(line.replace('# TODO:', '')))
                        else:
                            already_todo = False
                            doc.write('{!s}\n'.format(line.replace('#', '')))
            doc.write('\n')
