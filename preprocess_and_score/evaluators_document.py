"""
DOCUMENT EVALUATORS
    This is the module where we store all the evaluator functions for DOCUMENTS to be used in the pipeline.

SCHEMA:
def <NAME_OF_YOUR_FUNCTION>_evaluator(document: Document, <PARAMETER1>: <TYPE> = <DEFAULT_VALUE>, <PARAMETER2>: <TYPE> = <DEFAULT_VALUE>, ...) -> float:
    \"""
    <FUNCTION_DESCRIPTION>
        :param document: (type: Document) document for which score is calculated.
        :param <PARAMETER1>: (type: <TYPE>, default: <DEFAULT_VALUE>) <DESCRIPTION>
        :param <PARAMETER2>: (type: <TYPE>, default: <DEFAULT_VALUE>) <DESCRIPTION>
        ...
    \"""
    <YOUR CODE>
    return <SCORE>

NOTE:
    The <SCORE> should be a value between 0 and 1 where 1 means the document is perfect (in the meanings of this score function
    context), and 0 implies we want to discard the document.

CAREFUL!! DO NOT MODIFY OR REMOVE EXISTING FUNCTIONS!
WHAT IF I want to modify one? Then, create a copy of it called <NAME_OF_YOUR_FUNCTION>_evaluator_v<X>, where <X> is the
    number of the version.
"""

from functools import partial
import inspect
import os
from typing import Callable, List, Tuple
from document import Paragraph, Document
import re
from distributions_pipeline import distributions
import preprocess_and_score.evaluators_utils as u
from functools import partial


# Function to convert text to dictionary
def text_to_dict(text):
    data = {}
    for line in text.splitlines():
        key, values_str = line.split(':')
        key = key.strip()
        values = [value.strip() for value in values_str.split(',')]
        data[key] = values
    return data


def remove_punctuation(input_string):
    punctuation = '''¡!"#$%&'()*+,-./:;<=>¿?@[\]^_`{|}~'''
    return ''.join([char for char in input_string if char not in punctuation])


def remove_num(input_string):
    return ''.join([char for char in input_string if not char.isdigit()])


def min_sentences_per_document_evaluator(document: Document, min_sentences: int = 4,
                                         missing_sentence_penalty: float = 0.7) -> float:
    """
    DISCARDED
    Returns a score that is reduced for each sentence less the document has with respect to a given minimum.
        :param document: (type: Document) document for which score is calculated.
        :param min_sentences: (type: int) minimum number of words considered an acceptable length.
        :param missing_sentence_penalty: (type: float) the amount of score reduction for each word less in
            the paragraph.
    """
    initial_score = 1
    n_missing_words = max(0, min_sentences - document.get_num_sentences())
    score = initial_score * missing_sentence_penalty ** n_missing_words
    return score


def min_sentences_per_document_evaluator_v2(document: Document, min_sentences: int = 4,
                                            missing_sentence_penalty: float = 0.7) -> float:
    """
    DISCARDED
    NOTE: This evaluator only applies if average_word_per_sentence_evaluator penalizes the document.
    Returns a score that is reduced for each sentence less the document has with respect to a given minimum.
        :param document: (type: Document) document for which score is calculated.
        :param min_sentences: (type: int) minimum number of sentences considered an acceptable length.
        :param missing_sentence_penalty: (type: float) the amount of score reduction for each word less in
            the paragraph.
    """
    average_word_per_sentence_score = average_word_per_sentence_evaluator(document)
    initial_score = 1

    if average_word_per_sentence_score != 1.0:
        n_missing_sentences = max(0, min_sentences - document.get_num_sentences())
        score = initial_score * missing_sentence_penalty ** n_missing_sentences

    else:
        score = initial_score

    return score


def min_words_per_document_evaluator(document: Document,
                                     ip: List[tuple] = distributions["min_words_per_document"]) -> float:
    """
    Returns a score that is reduced for each word less the document has with respect to a given minimum.
        :param document: (type: Document) document for which score is calculated.
        :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.
    """
    words = document.get_num_words()

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(words)

    return res


def lowercase_letters_evaluator(document: Document, max_non_lowercase_percentage: float = 0.3,
                                max_penalty: float = 0.4) -> float:
    """
    DISCARDED
    Returns a lower score if the percentage of non-lowercase chars exceeds a given threshold.
        :param document: (type: Document) document for which score is calculated.
        :param max_non_lowercase_percentage: (type: float) maximum threshold mentioned.
        :param max_penalty: (type: float) max achievable penalty ( if percentage is 1 ).
    """
    lowercase_count = sum(1 for paragraph in document for sentence in paragraph for char in sentence if char.islower())
    char_count = document.get_num_characters()
    non_lowercase_frequency = (char_count - lowercase_count) / char_count

    return max_penalty ** (max(0., (non_lowercase_frequency - max_non_lowercase_percentage)) /
                           (1 - max_non_lowercase_percentage))


def average_sentences_per_paragraph_evaluator(document: Document, min_average_sentences_per_paragraph: float = 1.5,
                                              max_penalty: float = 0.4) -> float:
    """
    DISCARDED
    CAREFUL: lists have one sentence per paragraph.
    Returns a lower score if average sentences per paragraph is too low.
        :param document: (type: Document) document for which score is calculated.
        :param min_average_sentences_per_paragraph: (type: float) minimum ratio accepted.
        :param max_penalty: (type: float) max achievable penalty ( if ratio is 1 ).
    """
    average_sentences_per_paragraph = document.get_num_sentences() / document.get_num_paragraphs()

    return max_penalty ** (max(0., (min_average_sentences_per_paragraph - average_sentences_per_paragraph)) /
                           (min_average_sentences_per_paragraph - 1))


def average_word_per_sentence_evaluator(document: Document,
                                        ip: List[tuple] = distributions["avg_words_per_sent"]) -> float:
    """
    Returns a lower score if average words per sentence is too low.
        :param document: (type: Document) document for which score is calculated.
        :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.
    """
    average_words_per_sentence = document.get_num_words() / document.get_num_sentences()

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(average_words_per_sentence)

    return res


def punctuation_ratio_evaluator(document: Document, punctuations: Tuple[str] = ('!', ',', '.', ':', ';', '?'),
                                min_punctuation_per_word: float = 1. / 20, max_penalty: float = 0.4) -> float:
    """
    CAREFUL: shares field with other char functions. May penalize too much those sentences / paragraphs.
    Returns a lower score if average punctuation / word is too low.
        :param document: (type: Document) document for which score is calculated.
        :param punctuations: (type: Tuple[str]) valid punctuation.
        :param min_punctuation_per_word: (type: float) minimum ratio.
        :param max_penalty: (type: float) max achievable penalty ( if ratio is 0 ).
    """
    punctuations_in_document = 0
    words = document.get_num_words()
    for paragraph in document:
        for sentence in paragraph:
            for char in sentence:
                if char in punctuations:
                    punctuations_in_document += 1
    punctuations_per_word = punctuations_in_document / words

    return max_penalty ** (max(0., (min_punctuation_per_word - punctuations_per_word)) / min_punctuation_per_word)


def punctuation_ratio_evaluator_v2(document: Document, punctuations: Tuple[str] = (
        '¡', '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '¿', '?',
        '@', '[', ']', '^', '_', '{', '|', '}', '~'),
                                   ip: List[tuple] = distributions["punctuation_ratio"]) -> float:
    """
    Version v2 upgrades:
        - added maximum threshold
        - expanded punctuation list
    CAREFUL: shares field with other char functions. May penalize too much those sentences/paragraphs.
    Returns a lower score if average punctuation / word is outside range.
        :param document: (type: Document) document for which score is calculated.
        :param punctuations: (type: Tuple[str]) valid punctuation.
        :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.
    """
    punctuations_in_document = 0
    words = document.get_num_words()

    for char in document.get_text():
        if char in punctuations:
            punctuations_in_document += 1

    punctuations_per_word = punctuations_in_document / words

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(punctuations_per_word)

    return res


def unique_sentences_ratio_evaluator(document: Document,
                                     ip: List[tuple] = distributions["unique_sentences_ratio"]) -> float:
    """
    CAREFUL: shares field with other char functions. May penalize too much those sentences / paragraphs
    Lower unique sentences ratio implies lower score
        :param document: (type: Document) document for which score is calculated.
        :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.
    """
    n_sentences = document.get_num_sentences()
    unique_sentences = set()
    for paragraph in document:
        for sentence in paragraph:
            unique_sentences.add(str(sentence))
    unique_sentences_ratio = len(unique_sentences) / n_sentences

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(unique_sentences_ratio)

    return res


'''
We already have the lyric exception, so we don't need a threshold and can go back to first version
def unique_sentences_ratio_evaluator_v2(document: Document, min_repeated_lines_ratio=0.5,
                                        max_penalty: float = 0.4) -> float:
    """
    CAREFUL: not poetry-friendly.
    CAREFUL: this shares field with other char functions. You may penalize too much those sentences / paragraphs
    Lower unique sentences ratio implies lower score
        :param document: (type: Document) document for which score is calculated.
        :param max_penalty: (type: float) max achievable penalty ( if ratio is 0 ).
        :param min_repeated_lines_ratio: (type: float) min percentage of repeated sentences.
    """
    n_sentences = document.get_num_sentences()
    unique_sentences = set()
    for paragraph in document:
        for sentence in paragraph:
            unique_sentences.add(str(sentence))
    unique_sentences_ratio = len(unique_sentences) / n_sentences
    repeated_sentences_ratio = 1 - unique_sentences_ratio

    # Set threshold to X% of repeated sentences
    if repeated_sentences_ratio >= min_repeated_lines_ratio:
        return max_penalty ** repeated_sentences_ratio
    else:
        return 1
'''

all_stopwords = None


def min_stopwords_per_document_evaluator(document: Document,
                                         ip: List[tuple] = distributions["min_stopwords_per_document"]) -> float:
    """
    Returns a lower score if it has less than X stopwords from the given language.
    :param document: (type: Document) document for which score is calculated.
    :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.
    """

    global all_stopwords
    if all_stopwords is None:
        with open('./preprocess_and_score/stopwords.txt', 'r', encoding='utf-8') as file:
            all_stopwords = text_to_dict(file.read())

    stopwords = all_stopwords.get(document.get_language(), [])

    if stopwords:
        num_stopwords = sum(1 for word in document.get_words() if word.lower() in stopwords)

        # Set a dynamic threshold where 20% of the words must be stopwords
        min_stopwords = round(int(0.1 * document.get_num_words()))

        # Difference between the actual and required number of stopwords
        missing_stopwords = max(0, min_stopwords - num_stopwords)

        inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
        res = inter_(missing_stopwords)

        return res
    else:
        return 1


def alnum_evaluator_v2(document: Document, max_non_alphanumeric_chars_ratio: float = 0.2, max_penalty: float = 0.01):
    """
    DISCARDED
    Version v2 upgrades:
        - no longer includes whitespaces as non-alphanumeric
        - added threshold for which no penalty is given
        - before applied at sentence level
    Returns a lower score if the sentence have more non-alphanumeric characters than expected.
        :param document: (type: Document) document for which score is calculated.
        :param max_non_alphanumeric_chars_ratio: (type: float) max ratio accepted.
        :param max_penalty: (type: bool) maximum penalty (if ratio = 1)
    """

    non_alphanumeric_chars = 0
    char_count = document.get_num_characters()

    for paragraph in document:
        for sentence in paragraph:
            for char in sentence:
                if not (char.isalnum() or char == " "):
                    non_alphanumeric_chars += 1

    non_alphanumeric_chars_ratio = non_alphanumeric_chars / char_count

    return max_penalty ** (max(0., (non_alphanumeric_chars_ratio - max_non_alphanumeric_chars_ratio)) /
                           (1 - max_non_alphanumeric_chars_ratio))


def brunet_index_evaluator(document: Document,
                           ip: List[tuple] = distributions["brunets_index"]):
    """
    Lower is better. 
    Overlaps with max_word_repetition_ratio_evaluator.
    W = N^(V^-0.165)
    W: brunets index
    N: total text length
    V: total vocabulary , i.e. number of unique words
    :param document: (type: Document) document for which score is calculated.
    :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.
    """

    n = document.get_num_words()
    clean_doc = remove_num(remove_punctuation(document.get_text().lower()))
    v = len(set(clean_doc.split()))

    try:
        w = n ** (v ** -0.165)
    except ZeroDivisionError:
        return 1

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(w)

    return res


def max_word_repetition_ratio_evaluator(document: Document, max_word_repetition_ratio: float = 0.02,
                                        max_penalty: float = 0.01):
    """
    DISCARDED
    Returns a lower score if the character ratio of the most frequent word is over the threshold.
    Better to use brunet_index_evaluator since this penalizes short documents.
    :param document: (type: Document) document for which score is calculated.
    :param max_word_repetition_ratio: (type: float) max character ratio of the most frequent word.
    :param max_penalty: (type: float) max achievable penalty (if ratio is just above 1).
    """

    n_chars = document.get_num_characters()
    words = document.get_words()
    # Filter out strings with only punctuation and stopwords
    global all_stopwords
    if not all_stopwords:
        with open('./preprocess_and_score/stopwords.txt', 'r', encoding='utf-8') as file:
            all_stopwords = text_to_dict(file.read())
    stopwords = all_stopwords.get(document.get_language(), [])
    words = [remove_punctuation(word).lower() for word in words
             if remove_punctuation(word) != ""
             and word.lower() not in stopwords]

    if words:
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1  # dog:1, house:3, cat:2

        frequent_words = [word for word, freq in word_freq.items() if freq > 1]  # house, cat

        if frequent_words:
            most_freq_word = max(frequent_words, key=lambda k: word_freq[k])  # house
            # Calculate ratio
            n_chars_most_freq_word = sum(len(word) for word in words if word == most_freq_word)  # 5x3=15
            most_freq_word_ratio = n_chars_most_freq_word / n_chars  # 15/24=0.625
        else:
            most_freq_word_ratio = 0
    else:
        most_freq_word_ratio = 0

    return max_penalty ** (max(0., (most_freq_word_ratio - max_word_repetition_ratio)) /
                           (1 - max_word_repetition_ratio))


def bad_language_id_evaluator(document: Document,
                              lang_priorities: List[str] = ["ca", "eu", "en", "es", "gl"], # TODO: Synchronize with language priorities
                              ip: List[tuple] = distributions["bad_language_id"]):
    """
    Based on the percentage of undesired languages, it identifies
    documents with potentially poor language identification.

    :param document: (type: Document) document for which score is calculated.
    :param lang_priorities: (type: List[str]) List of ISO-639-1 codes for the desired languages in the document
    :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.

    Returns:
        float: A penalty score representing the likelihood of bad language identification in the document.
            A higher penalty indicates a higher likelihood of inaccurate language identification.
    """

    undesired_lang_ratio = 0
    lang_dict = document.get_languages()

    for lang, score in lang_dict.items():
        if lang not in lang_priorities:
            undesired_lang_ratio += score

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(undesired_lang_ratio)

    return res


def is_cursed(sentence):
    if any(curse in sentence for curse in cursed_substrings):
        return True
    if any(regex.findall(sentence) for regex in cursed_regex):
        return True
    else:
        return False


cursed_substrings = None
cursed_regex = None


def cursed_regex_evaluator(document: Document,
                           ip: List[tuple] = distributions["cursed_regex"]):
    """
    Calculate a combined penalty score for a document based on cursed patterns.

    This function analyzes the input document for the presence of "cursed" patterns
    using regular expressions. It calculates the ratio of lines in the document with
    "cursed" patterns and creates a penalty score based on a threshold.

    :param document: A Document object representing the input document for evaluation.
    :param ip: (type: List[tuple]) list of interpolation values to calculate the score function.
    :return: A combined penalty score reflecting the presence of "cursed" patterns in the document.
    """

    global cursed_substrings
    global cursed_regex
    if not cursed_substrings or not cursed_regex:
        # Substrings which don't need to compile
        cursed_substrings = [
            ' \u2116', '\ufffd\ufffd\ufffd', '\\|\\s*$', ' nr\\.$',
            'aute irure dolor ', ' sunt in culpa qui ', 'orem ipsum ', ' quis nostrud ',
            ' adipisicing ', ' dolore eu ', ' cupidatat ', 'autem vel eum', 'wisi enim ad',
            ' sex ', ' porn ', '\u9ec4\u8272\u7535\u5f71', 'mp3', 'ownload',
            'Vol\\.', ' Ep\\.', 'Episode', ' \u0433\\.\\s*$', ' \u043a\u0433\\.\\s*$',
            ' \u0448\u0442\\.', 'Develop', 'Facebook', ' crusher ', ' xxx ',
        ]
        # Regex which need to compile
        cursed_patterns = [
            ' ... ... ... ... ... ... ... ... ...',
            ' .... .... .... .... .... .... .... .... ....',
            ' [^ ] [^ ] [^ ] [^ ] [^ ] [^ ] [^ ] [^ ] [^ ]',
            ', ..,,? ..,,? ..,,? ..,,?',
        ]
        cursed_regex = [re.compile(pattern) for pattern in cursed_patterns]

    cursed_lines_count = 0
    # Count lines with cursed patterns
    for para in document:
        for sent in para:
            if is_cursed(sent):
                cursed_lines_count += 1

    # Calculate the ratio of cursed lines
    total_lines = document.get_num_sentences()
    cursed_lines_ratio = cursed_lines_count / total_lines if total_lines > 0 else 0.0

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(cursed_lines_ratio)

    return res


def wrong_begin_end_line_ratio_evaluator(document: Document,
                               ip: List[tuple] = distributions["bad_bol_eol_ratio"]):

    banned_begin_chars = ['.', '?', '!', ':', ',', ';', '=', '&', '%', '/', '\\', '~', '`', '§', '@', '|', '¤']
    allowed_end_chars = ['.', ';', '!', '…', '?', ')', ']', '}', '"', "'"]

    bad_line_count = 0

    average_words_per_sentence = document.get_num_words() / document.get_num_sentences()

    for parag in document:
        for sentence in parag:
            if sentence.get_num_words() >= average_words_per_sentence:
                sentence = sentence.strip()
                # Check if the line starts with a rare punctuation mark
                if sentence:
                    # Check if the sentence has at least two characters
                    if len(sentence) >= 2:
                        if (sentence[1].islower() or sentence[1] in banned_begin_chars) or (
                                not sentence[-1].isalnum() and sentence[-1] not in allowed_end_chars):
                            bad_line_count += 1

    # Calculate the ratio
    wrong_begin_end_line_ratio = bad_line_count / document.get_num_sentences()

    inter_ = partial(u.interpolation, [(val, 1 - score) for val, score in ip])
    res = inter_(wrong_begin_end_line_ratio)

    return res


# ---------------------------------------------------------------------------
# BSC-EDU classifier
# ---------------------------------------------------------------------------

_BSCEDU_MODEL = None
_BSCEDU_TOKENIZER = None

_BSCEDU_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "bsc-edu-classifier")
_BSCEDU_HF_REPO = "BSC-LT/bsc-edu-classifier"

# The model returns a regression score in the range [-2, 4].
# We normalise it to [0, 1] for compatibility with the rest of the pipeline.
_BSCEDU_SCORE_MIN = -2.0
_BSCEDU_SCORE_MAX = 4.0


def _load_bscedu_model():
    """Lazy-load the BSC-EDU classifier model and tokeniser.

    The model is first looked up in ``preprocess_and_score/models/bsc-edu-classifier``.
    If it is not found there, it is downloaded from HuggingFace
    (``BSC-LT/bsc-edu-classifier``) and saved in that directory.
    """
    global _BSCEDU_MODEL, _BSCEDU_TOKENIZER

    if _BSCEDU_MODEL is not None:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "The bscedu_document_evaluator requires 'torch' and 'transformers'. "
            "Install them with: pip install torch transformers"
        ) from exc

    model_path = _BSCEDU_MODEL_DIR

    if not os.path.isdir(model_path) or not os.listdir(model_path):
        print(f"[bscedu] Model not found at '{model_path}'. Downloading from HuggingFace ({_BSCEDU_HF_REPO})…")
        os.makedirs(model_path, exist_ok=True)
        # snapshot_download saves all model files locally
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=_BSCEDU_HF_REPO, local_dir=model_path)
        except Exception:
            # Fall back to from_pretrained with cache_dir
            tokenizer_tmp = AutoTokenizer.from_pretrained(_BSCEDU_HF_REPO, use_fast=True)
            model_tmp = AutoModelForSequenceClassification.from_pretrained(_BSCEDU_HF_REPO)
            tokenizer_tmp.save_pretrained(model_path)
            model_tmp.save_pretrained(model_path)
        print(f"[bscedu] Model saved to '{model_path}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    _BSCEDU_TOKENIZER = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    _BSCEDU_MODEL = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=dtype)
    _BSCEDU_MODEL.to(device)
    _BSCEDU_MODEL.eval()
    print(f"[bscedu] Model loaded on {device}.")


def bscedu_document_evaluator(document: Document) -> float:
    """
    Assigns a quality score to a document using the BSC-EDU regression classifier
    (``BSC-LT/bsc-edu-classifier``).

    The model produces a continuous score in the range **[-2, 4]**, where higher
    values indicate higher educational / quality value.  This raw score is
    linearly normalised to **[0, 1]** before being returned, so that it is
    compatible with the rest of the pipeline's scoring framework.

    The model is a document-level evaluator: the full text of the document is
    tokenised (truncated to 512 sub-word tokens) and fed to the model in one
    pass.  GPU inference is used automatically when CUDA is available; otherwise
    the model runs on CPU with float32 precision.

    :param document: (type: Document) document for which the quality score is
        calculated.
    """
    if document is None:
        return 0.0

    _load_bscedu_model()

    import torch

    text = document.get_text()
    if not text or not text.strip():
        return 0.0

    device = next(_BSCEDU_MODEL.parameters()).device

    inputs = _BSCEDU_TOKENIZER(
        text,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = _BSCEDU_MODEL(**inputs)
        logits = outputs.logits
        if logits.shape[-1] == 1:
            raw_score = logits.squeeze(-1).item()
        else:
            raw_score = torch.softmax(logits, dim=-1)[0, 1].item()

    # Normalise from [-2, 4] to [0, 1]
    normalised = (raw_score - _BSCEDU_SCORE_MIN) / (_BSCEDU_SCORE_MAX - _BSCEDU_SCORE_MIN)
    return float(max(0.0, min(1.0, normalised)))