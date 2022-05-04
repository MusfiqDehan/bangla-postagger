from deep_translator import GoogleTranslator


# Digit Translation
digit_converter = {
    '০': '0',
    '১': '1',
    '২': '2',
    '৩': '3',
    '৪': '4',
    '৫': '5',
    '৬': '6',
    '৭': '7',
    '৮': '8',
    '৯': '9'
}


def get_translated_digit(sentence):
    translated_sentence = []
    for each_letter in sentence:
        if each_letter in digit_converter.keys():
            translated_sentence.append(digit_converter[each_letter])
            # print(digit_converter[each_letter], end="")
        else:
            translated_sentence.append(each_letter)
            # print(each_letter, end="")

    return "".join(each for each in translated_sentence)


# ========================
# Translate a sentence
# ========================
def get_translation(sentence: str, source="bn", target="en") -> str:
    """
    Translate a sentence from one language to another using Google Translator.\n
    At first install dependencies \n
    `!pip install -U deep-translator`

    """
    translator = GoogleTranslator()
    translated_sentence = translator.translate(
        sentence, source=source, target=target)
    return translated_sentence
