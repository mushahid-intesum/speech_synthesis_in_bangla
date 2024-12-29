from copy import deepcopy
import torch
import dgl
import networkx as nx
from model.dependency_parser import BengaliDependencyParser

class Sentence2GraphParser:
    
    def __init__(self, language='bn', use_gpu=False):
        self.language = language
        self.parser = BengaliDependencyParser()
        # self.stanza_parser = stanza.Pipeline(lang=language, use_gpu=use_gpu)

    def parse(self, clean_sentence=None, words=None, ph_words=None):
        if self.language == 'zh' or self.language == 'bn':
            assert words is not None and ph_words is not None
            ret = self._parse_bn(words, ph_words)
        else:
            raise NotImplementedError
        return ret

    def _parse_bn(self, words, ph_words, enable_backward_edge=True, enable_recur_edge=True,
                  enable_inter_sentence_edge=True, sequential_edge=False):
        """
        words: <List of str>, each character in chinese is one item
        ph_words: <List of str>, each character in chinese is one item, represented by the phoneme
        Example:
                text1 = '宝马配挂跛骡鞍,貂蝉怨枕董翁榻.'
                words = ['<BOS>', '宝', '马', '配', '挂', '跛', '骡', '鞍', ','
                        , '貂', '蝉', '怨', '枕', '董', '翁', '榻', '<EOS>']
                ph_words = ['<BOS>', 'b_ao3_|', 'm_a3_#', 'p_ei4_|', 'g_ua4_#',
                            'b_o3_#', 'l_uo2_|', 'an1', ',', 'd_iao1_|',
                            'ch_an2_#', 'van4_#', 'zh_en3_#', 'd_ong3_|', 'ueng1_#', 't_a4', '<EOS>']
        """
        # words, ph_words = words[1:-1], ph_words[1:-1]  # delete <BOS> and <EOS>
        tmp_words = deepcopy(words)
        clean_text = ''.join(tmp_words).strip()

        parser_out = self.parser.parse(clean_text)

        idx_to_word = {i + 1: w for i, w in enumerate(words)}

        vocab_nodes = {}
        vocab_idx_offset = 0
        for sentence in parser_out.sentences:
            num_nodes_in_current_sentence = 0
            for vocab_node in sentence.words:
                num_nodes_in_current_sentence += 1
                vocab_idx = vocab_node.id + vocab_idx_offset
                vocab_text = vocab_node.text.replace(" ", "")  # delete blank in vocab
                vocab_nodes[vocab_idx] = vocab_text
            vocab_idx_offset += num_nodes_in_current_sentence

        # start vocab-to-word alignment
        vocab_to_word = {}
        current_word_idx = 1
        for vocab_i in vocab_nodes.keys():
            vocab_to_word[vocab_i] = []
            for w_in_vocab_i in vocab_nodes[vocab_i]:
                # if w_in_vocab_i != idx_to_word[current_word_idx]:
                #     raise ValueError("Word Mismatch!")
                vocab_to_word[vocab_i].append(current_word_idx)  # add a path (vocab_node_idx, word_global_idx)
                current_word_idx += 1

        # then we compute the vocab-level edges
        if len(parser_out.sentences) > 5:
            print("Detect more than 5 input sentence! pls check whether the sentence is too long!")
        vocab_level_source_id, vocab_level_dest_id = [], []
        vocab_level_edge_types = []
        sentences_heads = []
        vocab_id_offset = 0
        # get forward edges
        for s in parser_out.sentences:
            for w in s.words:
                w_idx = w.id + vocab_id_offset  # it starts from 1, just same as binarizer
                w_dest_idx = w.head + vocab_id_offset
                if w.head == 0:
                    sentences_heads.append(w_idx)
                    continue
                vocab_level_source_id.append(w_idx)
                vocab_level_dest_id.append(w_dest_idx)
            vocab_id_offset += len(s.words)
        vocab_level_edge_types += [0] * len(vocab_level_source_id)
        num_vocab = vocab_id_offset

        # optional: get backward edges
        if enable_backward_edge:
            back_source, back_dest = deepcopy(vocab_level_dest_id), deepcopy(vocab_level_source_id)
            vocab_level_source_id += back_source
            vocab_level_dest_id += back_dest
            vocab_level_edge_types += [1] * len(back_source)

        # optional: get inter-sentence edges if num_sentences > 1
        inter_sentence_source, inter_sentence_dest = [], []
        if enable_inter_sentence_edge and len(sentences_heads) > 1:
            def get_full_graph_edges(nodes):
                tmp_edges = []
                for i, node_i in enumerate(nodes):
                    for j, node_j in enumerate(nodes):
                        if i == j:
                            continue
                        tmp_edges.append((node_i, node_j))
                return tmp_edges

            tmp_edges = get_full_graph_edges(sentences_heads)
            for (source, dest) in tmp_edges:
                inter_sentence_source.append(source)
                inter_sentence_dest.append(dest)
            vocab_level_source_id += inter_sentence_source
            vocab_level_dest_id += inter_sentence_dest
            vocab_level_edge_types += [3] * len(inter_sentence_source)

        if sequential_edge:
            seq_source, seq_dest = list(range(1, num_vocab)) + list(range(num_vocab, 0, -1)), \
                                   list(range(2, num_vocab + 1)) + list(range(num_vocab - 1, -1, -1))
            vocab_level_source_id += seq_source
            vocab_level_dest_id += seq_dest
            vocab_level_edge_types += [4] * (num_vocab - 1) + [5] * (num_vocab - 1)

        # Then, we use the vocab-level edges and the vocab-to-word path, to construct the word-level graph
        num_word = len(words)
        source_id, dest_id, edge_types = [], [], []

        # print(vocab_level_source_id)
        # print(vocab_level_dest_id)
        # print(vocab_level_edge_types)
        for (vocab_start, vocab_end, vocab_edge_type) in zip(vocab_level_source_id, vocab_level_dest_id,
                                                             vocab_level_edge_types):

            if vocab_end in vocab_to_word.keys() and vocab_start in vocab_to_word.keys():
                # connect the first word in the vocab
                word_start = min(vocab_to_word[vocab_start])
                word_end = min(vocab_to_word[vocab_end])
                source_id.append(word_start)
                dest_id.append(word_end)
                edge_types.append(vocab_edge_type)

        # sequential connection in words
        for word_indices_in_v in vocab_to_word.values():
            for i, word_idx in enumerate(word_indices_in_v):
                if i + 1 < len(word_indices_in_v):
                    source_id.append(word_idx)
                    dest_id.append(word_idx + 1)
                    edge_types.append(4)
                if i - 1 >= 0:
                    source_id.append(word_idx)
                    dest_id.append(word_idx - 1)
                    edge_types.append(5)

        # optional: get recurrent edges
        if enable_recur_edge:
            recur_source, recur_dest = list(range(1, num_word + 1)), list(range(1, num_word + 1))
            source_id += recur_source
            dest_id += recur_dest
            edge_types += [2] * len(recur_source)

        # add <BOS> and <EOS>
        source_id += [0, num_word + 1, 1, num_word]
        dest_id += [1, num_word, 0, num_word + 1]
        edge_types += [4, 4, 5, 5]  # 4 represents sequentially forward, 5 is sequential backward

        edges = (torch.LongTensor(source_id), torch.LongTensor(dest_id))
        dgl_graph = dgl.graph(edges)
        assert dgl_graph.num_edges() == len(edge_types)
        return dgl_graph, torch.LongTensor(edge_types)


def plot_dgl_sentence_graph(dgl_graph, labels):
    """
    labels = {idx: word for idx,word in enumerate(sentence.split(" ")) }
    """
    import matplotlib.pyplot as plt
    nx_graph = dgl_graph.to_networkx()
    pos = nx.random_layout(nx_graph)
    nx.draw(nx_graph, pos, with_labels=False)
    nx.draw_networkx_labels(nx_graph, pos, labels)
    plt.show()

if __name__ == '__main__':

    # Unit Test for Chinese Graph Builder
    parser = Sentence2GraphParser("zh")
    text1 = '宝马配挂跛骡鞍,貂蝉怨枕董翁榻.'
    words = ['<BOS>', '宝', '马', '配', '挂', '跛', '骡', '鞍', ',', '貂', '蝉', '怨', '枕', '董', '翁', '榻', '<EOS>']
    ph_words = ['<BOS>', 'b_ao3_|', 'm_a3_#', 'p_ei4_|', 'g_ua4_#', 'b_o3_#', 'l_uo2_|', 'an1', ',', 'd_iao1_|',
                'ch_an2_#', 'van4_#', 'zh_en3_#', 'd_ong3_|', 'ueng1_#', 't_a4', '<EOS>']
    graph1, etypes1 = parser.parse(text1, words, ph_words)
    plot_dgl_sentence_graph(graph1, {i: w for i, w in enumerate(ph_words)})

    # Unit Test for English Graph Builder
    parser = Sentence2GraphParser("bn")
    text2 = "আধ ঘন্টার মধ্যে খলিল ভাই একজন হেকিম নিয়ে উপস্থিত হলেন।"
    graph2, etypes2 = parser.parse(text2)
    plot_dgl_sentence_graph(graph2, {i: w for i, w in enumerate(("<BOS> " + text2 + " <EOS>").split(" "))})
    
