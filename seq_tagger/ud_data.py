import os

from torchtext import data

lang_map = {'af':   'Afrikaans',
            'ar':   'Arabic',
            'be':   'Belarusian',
            'bg':   'Bulgarian',
            'cs':   'Czech',
            'en':   'English',
            'et':   'Estonian',
            'etn': 'EstonianNltk',
            'etn2': 'EstonianNltk2',
            'eu':   'Basque',
            'kk':   'Kazakh',
            'ro':   'Romanian'}


class SequenceTaggingDataset(data.Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.
    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]
    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, logger, separator='\t'):
        examples = []
        columns = []
        logger.info('input path: {}'.format(path))
        with open(path) as input_file:
            print('columns')
            for line in input_file:
                line = line.strip()
                # This condition was added to torchtext SequenceTaggingDataset class
                # to account for comment lines in input files
                if line.startswith('#'):
                    continue
                elif line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    print(columns)
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)
                    print(columns)
            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(SequenceTaggingDataset, self).__init__(examples, fields)


class UDPOS(SequenceTaggingDataset):

    # Universal Dependencies dataset

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None, test=None, **kwargs):
        cls.name = 'ud-treebanks-v2.1'
        cls.dirname = 'UD_{}'.format(lang_map[kwargs['lang']])
       #  print(root, cls.name, cls.dirname)

        path = os.path.join(root, cls.name, cls.dirname)

        return super(UDPOS, cls).splits(path=path, root=root, train=train,
                                             validation=validation,
                                             test=test,
                                             fields=kwargs['fields'], logger=kwargs['logger'])


class MorphTaggingDataset(data.Dataset):
    """Defines a dataset morphological tagging read from
    a CONLLU file. The POS and morph fields must be concatenated.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, logger, separator='\t'):
        examples = []
        cols = []
        logger.info('input path: {}'.format(path))
        with open(path) as input_file:
            morphs = {}
            for line in input_file:
                line = line.strip()
                # This condition was added to torchtext SequenceTaggingDataset class
                # to account for comment lines in input files
                if line.startswith('#'):
                    continue
                elif line == "":
                    if cols:
                        columns = [cols[2], [], []]
                        for w, pos, morph in zip(cols[1], cols[4], cols[5]):
                            posmorph = pos + '|' + morph
                            columns[1].append(posmorph)
                            columns[2].append(w)
                            # if w in morphs:
                            #     morphs[posmorph].add(w)
                            # else:
                            #     morphs[posmorph] = set()
                            #     morphs[posmorph].add(w)
                        # print(columns)
                        examples.append(data.Example.fromlist(columns, fields))
                    cols = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(cols) < i + 1:
                            cols.append([])
                        cols[i].append(column)
            print(morphs)
            # print("cols")
            # print(examples)
            # print(fields)
            # print(cols)
            if cols:
                print(cols)
                columns = [cols[1], []]
                for pos, morph in zip(cols[4], cols[5]):
                    columns[1].append(pos + '|' + morph)
                examples.append(data.Example.fromlist(columns, fields))
        super(MorphTaggingDataset, self).__init__(examples, fields)


class UDPOSMorph(MorphTaggingDataset):

    # Universal Dependencies dataset

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None, test=None, **kwargs):
        cls.name = 'ud-treebanks-v2.1'
        cls.dirname = 'UD_{}'.format(lang_map[kwargs['lang']])
       #  print(root, cls.name, cls.dirname)

        path = os.path.join(root, cls.name, cls.dirname)

        return super(UDPOSMorph, cls).splits(path=path, root=root, train=train,
                                             validation=validation,
                                             test=test,
                                             fields=kwargs['fields'], logger=kwargs['logger'])