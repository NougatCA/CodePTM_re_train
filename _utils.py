import json
import random

import torch
from torch.utils.data.dataset import Dataset
import pickle
import os
import gc


def remove_special_tokens(s, tokenizer):
    for token in tokenizer.all_special_tokens:
        s = s.replace(token, " ")
    return s


def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    elif args.model_type == 'gpt2':
        source_ids = tokenizer.encode(example.source, max_length=args.max_source_length, truncation=True)
        source_txt = tokenizer.decode(source_ids, skip_special_tokens=True)
        if stage == 'train':
            source_str = f"{source_txt} <|seq2seq_separator|> {example.target}"
        else:
            source_str = f"{source_txt} <|seq2seq_separator|>"
    else:
        source_str = example.source

    source_str = remove_special_tokens(source_str, tokenizer)
    source_ids = tokenizer.encode(source_str,
                                  max_length=args.max_source_length + args.max_target_length if args.model_type == "gpt2" and stage == "train" else args.max_source_length,
                                  padding='max_length',
                                  truncation=True)

    # print(source_str)
    # print(source_ids)

    if args.model_type != 'gpt2':
        assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    elif args.model_type == 'gpt2':
        target_ids = torch.Tensor(source_ids).long()
    else:
        target_str = example.target if args.task != "exception" else example.target_txt.lower()
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone', "qa"]:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url if hasattr(example, "url") else None
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


def convert_exception_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return ExceptionInputFeatures(example_id=example_index, source_ids=code, label=example.target)


def convert_qa_examples_to_features(item):
    example, example_index, tokenizer, args = item
    source = remove_special_tokens(example.source, tokenizer)
    nl = remove_special_tokens(example.nl, tokenizer)

    code = tokenizer.encode(source, max_length=args.max_source_length, padding='max_length', truncation=True)
    nl = tokenizer.encode(nl, max_length=args.max_target_length, padding='max_length', truncation=True)
    source_ids = code + nl
    return QAInputFeatures(example_index, source_ids, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class QAInputFeatures(object):
    def __init__(self, example_id, source_ids, label):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class ExceptionInputFeatures(object):

    def __init__(self, example_id, source_ids, label):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None,
                 source_txt=None,
                 target_txt=None,
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class ExceptionExample(object):

    def __init__(self, idx, source, target, target_txt, info):
        self.idx = idx
        self.source = source
        self.target = target
        self.target_txt = target_txt
        self.info = info


class QAExample(object):
    def __init__(self, idx, code, nl, target, info=None):
        self.idx = idx
        self.source = code
        self.nl = nl
        self.target = target
        self.info = info


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_assert_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip().lower(),
                    target=line2.strip().lower(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_mutant_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip().lower(),
                    target=line2.strip().lower(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_exception_examples(filename, data_num, type_list: list):
    assert type_list is not None and len(type_list) != 0
    examples = []
    with open(filename, encoding="utf-8") as f:
        for index, line in enumerate(f):
            js = json.loads(line.strip())
            code = " ".join(js["function"].split())
            target_txt = js["label"].lower()
            examples.append(
                ExceptionExample(
                    idx=index,
                    source=code,
                    target=type_list.index(target_txt),
                    target_txt=target_txt,
                    info=js["info"]
                )
            )
            if index + 1 == data_num:
                break
    return examples


def read_qa_examples(filename, data_num):
    examples = []
    with open(filename, encoding="utf-8") as f:
        for index, line in enumerate(f):
            js = json.loads(line.strip())
            code = " ".join(js["function"].split())
            doc = " ".join(js["docstring"].split())
            target_txt = js["label"].lower()
            if target_txt == "correct":
                target = 1
            elif target_txt == "incorrect":
                target = 0
            else:
                continue
            examples.append(
                QAExample(
                    idx=index,
                    code=code,
                    nl=doc,
                    target=target,
                    info=js["info"]
                )
            )
            if index + 1 == data_num:
                break
    return examples


def read_cosqa_examples(filename, data_num):
    with open(filename, encoding='utf-8') as f:
        data_list = json.load(f)
    data = []
    idx = 0
    for json_data in data_list:
        if 'label' not in json_data:
            label = 0
        elif json_data['label'] == 0:
            label = 0
        else:
            label = 1
        nl = ' '.join(json_data['doc'].split())
        code = ' '.join(json_data['code'].split())
        data.append(QAExample(
            idx=json_data["idx"],
            code=code,
            nl=nl,
            target=label
        ))
        idx += 1
        if idx == data_num:
            break
    return data


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data

# examples = []
#     assert len(filename.split(',')) == 2
#     src_filename = filename.split(',')[0]
#     trg_filename = filename.split(',')[1]
#     idx = 0
#
#     with open(src_filename) as f1, open(trg_filename) as f2:
#         for line1, line2 in zip(f1, f2):
#             examples.append(
#                 Example(
#                     idx=idx,
#                     source=line1.strip().lower(),
#                     target=line2.strip().lower(),
#                 )
#             )
#             idx += 1
#             if idx == data_num:
#                 break
#     return examples

def read_completion_line_examples(filename: str, data_num):
    examples = []
    if filename.endswith(".txt"):
        idx = 0
        with open(filename, encoding="utf-8") as f:
            for line in f.readlines():
                line_tokens = line.strip().split()
                line_len = len(line_tokens)
                for _ in range(3):
                    pos = int(line_len * (random.random() * 0.7 + 0.15))
                    src = " ".join(line_tokens[:pos])
                    tgt = " ".join(line_tokens[pos:])
                    examples.append(Example(idx=idx,
                                            source=src,
                                            target=tgt))
                idx += 3
                if idx >= data_num:
                    break
    elif filename.endswith(".json"):
        idx = 0
        with open(filename, encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line.strip())
                examples.append(Example(idx=data["id"],
                                        source=data["input"],
                                        target=data["gt"]))
                idx += 1
                if idx == data_num:
                    break
    return examples


class CompletionDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type + "_blocksize_%d" % (block_size) + "_wordsize_%d" % (
            world_size) + "_rank_%d" % (local_rank))
        if os.path.exists(cached_file):
            if file_type == 'train':
                logger.warning("Loading features from cached file %s", cached_file)
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = args.train_filename
            if file_type == 'train':
                logger.warning("Creating features from dataset file at %s", datafile)
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d" % (length))
            input_ids = []
            for idx, x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("Rank %d, load %d" % (local_rank, percent))
            del data
            gc.collect()

            length = len(input_ids) // world_size
            logger.info(f"tokens: {length * world_size}")
            input_ids = input_ids[local_rank * length: (local_rank + 1) * length]

            for i in range(0, length - block_size, block_size):
                self.inputs.append(input_ids[i: i + block_size])
            del input_ids
            gc.collect()

            if file_type == 'train':
                logger.warning("Rank %d Training %d token, %d samples" % (local_rank, length, len(self.inputs)))
                logger.warning("Saving features into cached file %s", cached_file)
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])


class CompletionEvalDataset(Dataset):
    def __init__(self, tokenizer, args, logger, file_type='train', block_size=1024):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cached_file = os.path.join(args.output_dir, file_type + "_blocksize_%d" % (block_size))
        if os.path.exists(cached_file):
            with open(cached_file, 'rb') as handle:
                self.inputs = pickle.load(handle)

        else:
            self.inputs = []

            datafile = args.test_filename
            with open(datafile) as f:
                data = f.readlines()

            length = len(data)
            logger.info("Data size: %d" % (length))
            input_ids = []
            for idx, x in enumerate(data):
                x = x.strip()
                if x.startswith("<s>") and x.endswith("</s>"):
                    pass
                else:
                    x = "<s> " + x + " </s>"
                try:
                    input_ids.extend(tokenizer.encode(x))
                except Exception:
                    pass
                if idx % (length // 10) == 0:
                    percent = idx / (length // 10) * 10
                    logger.warning("load %d" % (percent))
            del data
            gc.collect()

            logger.info(f"tokens: {len(input_ids)}")
            self.split(input_ids, tokenizer, logger, block_size=block_size)
            del input_ids
            gc.collect()

            with open(cached_file, 'wb') as handle:
                pickle.dump(self.inputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def split(self, input_ids, tokenizer, logger, block_size=1024):
        sample = []
        i = 0
        while i < len(input_ids):
            sample = input_ids[i: i + block_size]
            if len(sample) == block_size:
                for j in range(block_size):
                    if tokenizer.convert_ids_to_tokens(sample[block_size - 1 - j])[
                        0] == '\u0120' or tokenizer.convert_ids_to_tokens(sample[block_size - 1 - j]).startswith(
                        "<NUM_LIT"):
                        break
                    if sample[block_size - 1 - j] in [tokenizer.bos_token_id, tokenizer.eos_token_id,
                                                      tokenizer.sep_token_id]:
                        if sample[block_size - 1 - j] != tokenizer.bos_token_id:
                            j -= 1
                        break
                if j == block_size - 1:
                    print(tokenizer.decode(sample))
                    exit()
                sample = sample[: block_size - 1 - j]
            # print(len(sample))
            i += len(sample)
            pad_len = block_size - len(sample)
            sample += [tokenizer.pad_token_id] * pad_len
            self.inputs.append(sample)

            if len(self.inputs) % 10000 == 0:
                logger.info(f"{len(self.inputs)} samples")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.tensor(self.inputs[item])
