import os

import jsonlines

def get_filename(path, allfile, dict_filetype=None):
    '''递归获得所有符合条件的文件名 
    
    @param : path 起始目录，要检查的根目录 
    @param : allfile 传入的初始文件名列表，填空即可
    @param : dict_filetype 要检查的文件类型，为None时则不检查返回所有。默认为None
    @return: 列表 所有与 dict_filetype 对应的文件名 
    '''
    filelist = os.listdir(path) 
    for filename in filelist: 
        filepath = os.path.join(path, filename) 
        # 判断文件夹 
        if os.path.isdir(filepath): 
            # 文件夹继续递归 
            get_filename(filepath, allfile, dict_filetype) 
        else: 
            temp_file_type = filepath.split(".")[-1]
            # 判断文件类型
            if dict_filetype is None or temp_file_type in dict_filetype: 
                allfile.append(filepath) 
            # 展示所有未包含的文件 
            else: 
                print("the file is not include : %s" % filepath ) 
    return allfile     


def build_source_data(path, type):
    file_names = get_filename(path, [], 'jsonl')
    with open(type + '.src', 'w', encoding='utf-8'):
        pass
    with open(type + '.tgt', 'w', encoding='utf-8'):
        pass
    for filename in file_names:
        code_tmp = []
        doc_tmp = []
        with jsonlines.open(filename) as f:
            for obj in f:
                code_tokens = ' '.join([item.replace('\n', '').replace('\r', '') for item in obj['code_tokens']]).strip()
                docstring_tokens = ' '.join([item.replace('\n', '').replace('\r', '') for item in obj['docstring_tokens']]).strip()
                if len(obj['code_tokens']) != 0 and len(obj['docstring_tokens']) != 0 and \
                    code_tokens != '' and docstring_tokens != '':
                    code_tmp.append(code_tokens + '\n')
                    doc_tmp.append(docstring_tokens + '\n')
                else:
                    print('empty line')
        print('--------------')
        print(filename)
        print(len(code_tmp))
        print(len(doc_tmp))
        with open(type + '.src', 'a', encoding='utf-8') as f:
            f.writelines(code_tmp)
        with open(type + '.tgt', 'a', encoding='utf-8') as f:
            f.writelines(doc_tmp)

def test(name):
    with open(name + '.src', 'r', encoding='utf-8') as f:
        count1 = len(f.readlines())
    with open(name + '.tgt', 'r', encoding='utf-8') as f:
        count2 = len(f.readlines())
    print(name)
    print('src' + str(count1))
    print('tgt' + str(count2))


if __name__ == '__main__':
    # build_source_data('../train', 'train')
    # build_source_data('../test', 'test')
    # build_source_data('../valid', 'valid')
    test('train')
    test('test')
    test('valid')

