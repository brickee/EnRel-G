import os
import logging
logger = logging.getLogger(__name__)



columns = {0: 'ner', 1:'pos', 2:'dep', 3:'head'}
pos_map = {'X':0, 'ADJ':1, 'ADP':2, 'ADV':3, 'AUX':4, 'CONJ':5, 'CCONJ':6, 'DET':7, 'INTJ':8, 'NOUN':9, 'NUM':10, 'PART':11, 'PRON':12, 'PROPN':13, 'PUNCT':14, 'SCONJ':15, 'SYM':16, 'VERB':17}
dep_map = {'X':0, 'acl':1, 'acomp':2, 'advcl':3, 'advmod':4, 'agent':5, 'amod':6, 'appos':7, 'attr':8, 'aux':9, 'auxpass':10, 'case':11, 'cc':12, 'ccomp':13, 'compound':14, 'conj':15, 'cop':16,
           'csubj':17, 'csubjpass':18, 'dative':19, 'dep':20, 'det':21, 'dobj':22, 'expl':23, 'intj':24, 'mark':25, 'meta':26, 'neg':27, 'nn':28, 'nmod':29, 'npmod':30, 'nsubj':31,
           'nsubjpass':32, 'nummod':33, 'oprd':34, 'obj':35, 'obl':36, 'parataxis':37, 'pcomp':38, 'pobj':39, 'poss':40, 'preconj':41, 'prep':42, 'prt':43, 'punct':44, 'quantmod':45, 'relcl':46,
           'ROOT':47, 'xcomp':48, 'nmod:npmod':49, 'nmod:poss':50, 'npadvmod':51,'acl:relcl':52,'cc:preconj':53,'mwe':54,'predet':55, 'det:predet':56, 'subtok':57
           }
# csubj: clausal subject


from pyGAT.build_graph import build_graph,get_sen_embed,build_sim_graph, build_dist_graph,combine_embeds

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, tags = None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.tags = tags

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, pos=None, dep = None, head = None, adj_a=None, adj_f=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        # self.pos = pos
        # self.dep = dep
        # self.head = head
        self.adj_a = adj_a
        self.adj_f = adj_f




def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.strip().split(' ')
        sentence.append(splits[0])
        label.append(splits[1:])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.para.128plus.128sub.Base.SciPosDep.conll")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.para.128plus.128sub.Base.SciPosDep.conll")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.para.128plus.128sub.Base.SciPosDep.conll")), "test")

    def get_labels(self, data_dir): # last one has to be 'SEP' ！！！！！
        # TODO: check if O should be first!
        # return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        # print(self._read_tsv(os.path.join(data_dir, "labels.txt"))[0][0])
        label_list = ['O'] # make O to be in the first place
        label_list.extend([i.strip() for i in self._read_tsv(os.path.join(data_dir, "labels.txt"))[0][0][:-1]])
        label_list.extend(["[CLS]", "[SEP]"])
        # print(label_list)
        return label_list

    def _create_examples(self,lines,set_type):
        examples = []
        # print(lines)
        for i,(sentence,labels) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = [lbl[0] for lbl in labels]
            tags = {}
            for tag_i, tag in columns.items():
                if tag_i>0:
                    tags[tag]=[lbl[tag_i] for lbl in labels]
                    assert len(tags[tag]) == len(label)
            # print(labels, label, tags)
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label, tags= tags))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, gat_type):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}
    # print(label_map)

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        # print(tags)
        for i, word in enumerate(textlist):
            #TODO:单个tokenize word的结果可能和整个tokenize sentence不一样！
            # input_id 基于tokens(ntokens)建立。 因为有valid mask，所以对BERT本身没有影响。
            # 但是图是在只保留了valid id上的构建的，所以要回归原句子的id
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]

            if not len(token): # deal with special token
                tokens.append('?')
                labels.append(label_1)
                valid.append(1)
                label_mask.append(1)
            else:
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)


        assert labels == labellist # if not, try to modify tag processing
        # tag process
        # pos, dep, head = ([] for i in range(3))
        for tag_name, tag in example.tags.items():

            if tag_name == 'pos':
                pos = [pos_map[t] for t in tag]
            elif tag_name == 'dep':
                dep = [dep_map[t] for t in tag]
            elif tag_name == 'head':
                head = [int(t) for t in tag]

            assert len(tag) == len(labels)
        # print(pos,dep,head)
        if len(tokens) >= max_seq_length - 1:  # TODO: only remove longer part!
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
            # tag process
            pos = pos[0:(max_seq_length - 2)]
            dep = dep[0:(max_seq_length - 2)]
            head = head[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)  # label mask take the SEP and CLS into account
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])  # label_ids and label_mask include SEP and CLS; but not label.
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)

        # add [SEP][CLS] and pad tags for convenience (accord with lable_ids)
        # tag process
        pos.insert(0,0)
        pos.append(0)
        dep.insert(0,0)
        dep.append(0)
        head.insert(0,-1)  # 0 is used in the head number!
        head.append(-1)

            # print(tag)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
            # tag process
            pos.append(0)
            dep.append(0)
            head.append(-1)


        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
            # tag process
            pos.append(0)
            dep.append(0)
            head.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        # tag process

        assert len(pos)==len(dep)==len(head)==len(label_ids)==max_seq_length





        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("valid_ids: %s" % " ".join([str(x) for x in valid]))

            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))
            logger.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))
            # tag process

            logger.info("pos tag: %s" % (" ".join([str(i) for i in pos])))
            logger.info("dep tag: %s" % (" ".join([str(i) for i in dep])))
            logger.info("head tag: %s" % (" ".join([str(i) for i in head])))


        # construct graphs
        if gat_type:

            #TODO: the code before works on batch level, here need compatible
            # 检查如何验证自己构图在模型中也是对的，不只是预处理
            # print(ntokens)
            sentence = textlist
            if len(sentence) >= max_seq_length - 1:  #only remove longer part!
                sentence = sentence[0:(max_seq_length - 2)]
            sentence.insert(0,'[CLS]')
            sentence.append('[SEP]')
            # print(ntokens)
            adj_a, _, adj_f, _, _ = build_graph([sentence], max_len=max_seq_length, pos_ids=[pos], dep_ids=[dep], head=[head])
            if gat_type == 'A':
                adj_a = adj_a[0].tolist()
                adj_f = None
            elif gat_type == 'F':
                adj_a = None
                adj_f = adj_f[0].tolist()
            elif gat_type == 'AF':
                adj_a = adj_a[0].tolist()
                adj_f = adj_f[0].tolist()
        else:
            adj_a = None
            adj_f = None


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              # pos = pos,
                              # dep = dep,
                              # head = head,
                              adj_a = adj_a,
                              adj_f = adj_f
                              ))
    return features






def construct_graphs(input_ids,tokenizer, pos_ids, dep_ids, head, max_len, type):
    sentences = []
    for i, input_id in enumerate(input_ids):
        input_id1 = input_id.to('cpu').numpy()
        #TODO: find that this step will not correctly reconstruct the original tokens (e.g. g(-1)->'g', '(', '-', '1', ')')
        sub_tokens = tokenizer.convert_ids_to_tokens(input_id1)
        # decode does not work
        # text = tokenizer.decode(input_id)[0]
        # tokens1 = text.strip().split(' ')
        print(sub_tokens)
        tokens2 = subtokens2tokens(sub_tokens)
        print(tokens2)
        tokens2 = [tkn for tkn in tokens2 if tkn != '[PAD]']
        #TODO: try to verify this if the performance is not in line with intuition.
        # assert len(tokens2) == len(valid_input_ids[i]) == lengths[i]

        sentences.append(tokens2)
    pos_ids = pos_ids.to('cpu').numpy()
    dep_ids = dep_ids.to('cpu').numpy()
    head = head.to('cpu').numpy()

    if type == 'AF':
        adj_a, _, adj_f, _, _ = build_graph(sentences, max_len,pos_ids=pos_ids, dep_ids=dep_ids, head=head)
        return adj_a,adj_f
    elif type == 'A':
        adj_a, _, _, _, _ = build_graph(sentences, max_len,pos_ids=pos_ids, dep_ids=dep_ids, head=head)
        return adj_a,None
    elif type == 'F':
        _, _, adj_f, _, _ = build_graph(sentences, max_len,pos_ids=pos_ids, dep_ids=dep_ids, head=head)
        return None,adj_f




def write2file(examples,y_true , y_pred,file_name):
    with open(file_name, "w") as writer:
        for i,y_sen in enumerate(y_true):
            eg = examples[i].text_a.split(' ')
            for j,lbl in enumerate(y_sen):
                line = ' '.join([eg[j], lbl, y_pred[i][j]])
                writer.write(line)
                writer.write('\n')
            writer.write('\n')

def write2report(output_test_file, report):
    with open(output_test_file, "w") as writer:
        writer.write(report)



def subtokens2tokens(tokens):
    def is_subtoken(word):
        if word[:2] == "##":
            return True
        else:
            return False
    # tokens = ['why', 'isn', "##'", '##t', 'Alex', "##'", 'text', 'token', '##izing']
    restored_text = []
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i + 1) < len(tokens) and is_subtoken(tokens[i + 1]):
            restored_text.append(tokens[i] + tokens[i + 1][2:])
            if (i + 2) < len(tokens) and is_subtoken(tokens[i + 2]):
                restored_text[-1] = restored_text[-1] + tokens[i + 2][2:]
        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])
    return restored_text