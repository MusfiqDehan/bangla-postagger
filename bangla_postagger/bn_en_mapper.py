from . import en_postaggers as ep
import torch
import itertools
import transformers
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()


# ========================================
# Get Aligned Words
# ========================================
def get_alignment_mapping(source="", target="", model_path="bert-base-multilingual-cased"):
    """
    Get Aligned Words
    """
    model = transformers.BertModel.from_pretrained(model_path)
    tokenizer = transformers.BertTokenizer.from_pretrained(model_path)

    # pre-processing
    sent_src, sent_tgt = source.strip().split(), target.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [
        tokenizer.tokenize(word) for word in sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [
        tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)[
        'input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []

    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]

    sub2word_map_tgt = []

    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8

    threshold = 1e-3

    model.eval()

    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[
            2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[
            2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * \
            (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)

    align_words = set()

    for i, j in align_subwords:
        align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

    return sent_src, sent_tgt, align_words


# ========================================
# Get Word Aligned Mapping
# ========================================
def get_word_mapping(source="", target="", model_path="bert-base-multilingual-cased"):
    """
    Get Word Aligned Mapping Words
    """
    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_path=model_path)

    result = []

    for i, j in sorted(align_words):
        result.append(f'bn:({sent_src[i]}) -> en:({sent_tgt[j]})')

    return result


# ========================================
# Get Word Aligned Mapping
# ========================================
def get_word_index_mapping(source="", target="", model_path="bert-base-multilingual-cased"):
    """
    Get Word Aligned Mapping Index
    """
    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_path=model_path)

    result = []

    for i, j in sorted(align_words):
        result.append(f'bn:({i}) -> en:({j})')

    return result


# ========================================
# Get NLTK PoS Tags
# ========================================
def get_nltk_postag(source="", target="", model_path="bert-base-multilingual-cased"):
    """
    Get NLTK PoS Tags
    """
    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_path=model_path)

    nltk_postag_dict = ep.get_nltk_postag_dict(target=target)

    mapped_sent_src = []
    result = []

    for i, j in sorted(align_words):
        punc = r"""!()-[]{}।;:'"\,<>./?@#$%^&*_~"""

        if sent_src[i] in punc or sent_tgt[j] in punc:
            mapped_sent_src.append(sent_src[i])
            result.append(
                f'bn:({sent_src[i]}) -> en:({sent_tgt[j]}) -> tag:(PUNC)')
        else:
            mapped_sent_src.append(sent_src[i])
            result.append(
                f'bn:({sent_src[i]}) -> en:({sent_tgt[j]}) -> tag:({nltk_postag_dict[sent_tgt[j]]})')

    unks = list(set(sent_src).difference(set(mapped_sent_src)))

    for word in unks:
        result.append(
            f'bn:({word}) -> en:(N/A) -> tag:(UNK)')

    pos_accuracy = ((len(sent_src) - len(unks)) / len(sent_src)) * 100

    # pos_accuracy = ( (len(sent_src) - len(unks)) / len(sent_src) )

    # pos_accuracy = f"PoS Tagging Accuracy = {pos_accuracy:0.2%}"

    return sent_src, mapped_sent_src, unks, result, pos_accuracy


# ========================================
# Get Spacy PoS Tags
# ========================================
def get_spacy_postag(source="", target="", model_path="bert-base-multilingual-cased"):
    """
    Get Spacy PoS Tags
    """
    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_path=model_path)

    spacy_postag_dict = ep.get_spacy_postag_dict(target=target)

    mapped_sent_src = []
    result = []

    for i, j in sorted(align_words):
        punc = r"""!()-[]{}।;:'"\,<>./?@#$%^&*_~"""

        if sent_src[i] in punc or sent_tgt[j] in punc:
            mapped_sent_src.append(sent_src[i])
            result.append(
                f'bn:({sent_src[i]}) -> en:({sent_tgt[j]}) -> tag:(PUNC)')
        else:
            mapped_sent_src.append(sent_src[i])
            result.append(
                f'bn:({sent_src[i]}) -> en:({sent_tgt[j]}) -> tag:({spacy_postag_dict[sent_tgt[j]]})')

    unks = list(set(sent_src).difference(set(mapped_sent_src)))

    for word in unks:
        result.append(
            f'bn:({word}) -> en:(N/A) -> tag:(UNK)')

    pos_accuracy = ((len(sent_src) - len(unks)) / len(sent_src)) * 100

    # pos_accuracy = ( (len(sent_src) - len(unks)) / len(sent_src) )

    # pos_accuracy = f"PoS Tagging Accuracy = {pos_accuracy:0.2%}"

    return sent_src, mapped_sent_src, unks, result, pos_accuracy


# ========================================
# Get Flair PoS Tags
# ========================================
def get_flair_postag(source="", target="", model_path="bert-base-multilingual-cased"):
    """
    Get Flair PoS Tags
    """
    sent_src, sent_tgt, align_words = get_alignment_mapping(
        source=source, target=target, model_path=model_path)

    flair_postag_dict = ep.get_flair_postag_dict(target=target)

    mapped_sent_src = []
    result = []

    for i, j in sorted(align_words):
        punc = r"""!()-[]{}।;:'"\,<>./?@#$%^&*_~"""

        if sent_src[i] in punc or sent_tgt[j] in punc:
            mapped_sent_src.append(sent_src[i])
            result.append(
                f'bn:({sent_src[i]}) -> en:({sent_tgt[j]}) -> tag:(PUNC)')
        else:
            mapped_sent_src.append(sent_src[i])
            result.append(
                f'bn:({sent_src[i]}) -> en:({sent_tgt[j]}) -> tag:({flair_postag_dict[sent_tgt[j]]})')

    unks = list(set(sent_src).difference(set(mapped_sent_src)))

    for word in unks:
        result.append(
            f'bn:({word}) -> en:(N/A) -> tag:(UNK)')

    pos_accuracy = ((len(sent_src) - len(unks)) / len(sent_src)) * 100

    # pos_accuracy = ( (len(sent_src) - len(unks)) / len(sent_src) )

    # pos_accuracy = f"PoS Tagging Accuracy = {pos_accuracy:0.2%}"

    return sent_src, mapped_sent_src, unks, result, pos_accuracy
