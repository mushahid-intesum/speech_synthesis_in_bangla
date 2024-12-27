import re

import bangla
from bnnumerizer import numerize
from bnunicodenormalizer import Normalizer

# initialize
bnorm = Normalizer()

numbers = ['৯', '৭', '১', '৮', '২', '৩', '৪', '৫', '০', '৬']

number_mappings = {
    '৯': '9', 
    '৭': '7', 
    '১': '1', 
    '৮': '8', 
    '২': '2', 
    '৩': '3', 
    '৪': '4', 
    '৫': '5', 
    '০': '0', 
    '৬': '6'
}

numeric_words = {
    '.': 'দশমিক',
    '0': 'শূন্য',
    '1': 'এক',
    '01': 'এক',
    '2': 'দুই',
    '02': 'দুই',
    '3': 'তিন',
    '03': 'তিন',
    '4': 'চার',
    '04': 'চার',
    '5': 'পাঁচ',
    '05': 'পাঁচ',
    '6': 'ছয়',
    '06': 'ছয়',
    '7': 'সাত',
    '07': 'সাত',
    '8': 'আট',
    '08': 'আট',
    '9': 'নয়',
    '09': 'নয়',
    '10': 'দশ',
    '11': 'এগারো',
    '12': 'বার',
    '13': 'তের',
    '14': 'চৌদ্দ',
    '15': 'পনের',
    '16': 'ষোল',
    '17': 'সতের',
    '18': 'আঠার',
    '19': 'উনিশ',
    '20': 'বিশ',
    '21': 'একুশ',
    '22': 'বাইশ',
    '23': 'তেইশ',
    '24': 'চব্বিশ',
    '25': 'পঁচিশ',
    '26': 'ছাব্বিশ',
    '27': 'সাতাশ',
    '28': 'আঠাশ',
    '29': 'ঊনত্রিশ',
    '30': 'ত্রিশ',
    '31': 'একত্রিশ',
    '32': 'বত্রিশ',
    '33': 'তেত্রিশ',
    '34': 'চৌত্রিশ',
    '35': 'পঁয়ত্রিশ',
    '36': 'ছত্রিশ',
    '37': 'সাঁইত্রিশ',
    '38': 'আটত্রিশ',
    '39': 'ঊনচল্লিশ',
    '40': 'চল্লিশ',
    '41': 'একচল্লিশ',
    '42': 'বিয়াল্লিশ',
    '43': 'তেতাল্লিশ',
    '44': 'চুয়াল্লিশ',
    '45': 'পঁয়তাল্লিশ',
    '46': 'ছেচল্লিশ',
    '47': 'সাতচল্লিশ',
    '48': 'আটচল্লিশ',
    '49': 'ঊনপঞ্চাশ',
    '50': 'পঞ্চাশ',
    '51': 'একান্ন',
    '52': 'বায়ান্ন',
    '53': 'তিপ্পান্ন',
    '54': 'চুয়ান্ন',
    '55': 'পঞ্চান্ন',
    '56': 'ছাপ্পান্ন',
    '57': 'সাতান্ন',
    '58': 'আটান্ন',
    '59': 'ঊনষাট',
    '60': 'ষাট',
    '61': 'একষট্টি',
    '62': 'বাষট্টি',
    '63': 'তেষট্টি',
    '64': 'চৌষট্টি',
    '65': 'পঁয়ষট্টি',
    '66': 'ছেষট্টি',
    '67': 'সাতষট্টি',
    '68': 'আটষট্টি',
    '69': 'ঊনসত্তর',
    '70': 'সত্তর',
    '71': 'একাত্তর',
    '72': 'বাহাত্তর',
    '73': 'তিয়াত্তর',
    '74': 'চুয়াত্তর',
    '75': 'পঁচাত্তর',
    '76': 'ছিয়াত্তর',
    '77': 'সাতাত্তর',
    '78': 'আটাত্তর',
    '79': 'ঊনআশি',
    '80': 'আশি',
    '81': 'একাশি',
    '82': 'বিরাশি',
    '83': 'তিরাশি',
    '84': 'চুরাশি',
    '85': 'পঁচাশি',
    '86': 'ছিয়াশি',
    '87': 'সাতাশি',
    '88': 'আটাশি',
    '89': 'ঊননব্বই',
    '90': 'নব্বই',
    '91': 'একানব্বই',
    '92': 'বিরানব্বই',
    '93': 'তিরানব্বই',
    '94': 'চুরানব্বই',
    '95': 'পঁচানব্বই',
    '96': 'ছিয়ানব্বই',
    '97': 'সাতানব্বই',
    '98': 'আটানব্বই',
    '99': 'নিরানব্বই',
    '100': 'একশো',
}

units = {
    'koti': 'কোটি',
    'lokkho': 'লক্ষ',
    'hazar': 'হাজার',
    'sotok': 'শত',
    'ekok': '',
}   


import math

def input_sanitizer(number):
    if isinstance(number, float) or isinstance(number, int) or \
            isinstance(number, str):
        if isinstance(number, str):
            try:
                if "." in number:
                    number = float(number)
                else:
                    number = int(number)
            except ValueError:
                return None
        return number
    else:
        return None


def generate_segments(number):
    """
    Generating the unit segments such as koti, lokkho
    """
    segments = dict()
    segments['koti'] = math.floor(number/10000000)
    number = number % 10000000
    segments['lokkho'] = math.floor(number/100000)
    number = number % 100000
    segments['hazar'] = math.floor(number/1000)
    number = number % 1000
    segments['sotok'] = math.floor(number/100)
    number = number % 100
    segments['ekok'] = number

    return segments


def float_int_extraction(number):
    """
    Extracting the float and int part from the passed number. The first return
    is the part before the decimal point and the rest is the fraction.
    """
    _number = str(number)
    if "." in _number:
        return tuple([int(x) for x in _number.split(".")])
    else:
        return number, None


def whole_part_word_gen(segments):
    """
    Generating the bengali word for the whole part of the number
    """
    generated_words = ''
    for segment in segments:
        if segments[segment]:
            generated_words += numeric_words[str(segments[segment])] + \
                " " + units[segment] + " "

    return generated_words[:-1]


def fraction_to_words(fraction):
    """
    Generating bengali words for the part after the decimal point
    """
    generated_words = ""
    for digit in str(fraction):
        generated_words += numeric_words[digit] + " "
    return generated_words[:-1]

def parse_phone_or_id_number(text):
    phone_num = ""

    for char in text:
        bn_num_to_eng_num = number_mappings[char]
        phone_num += numeric_words[bn_num_to_eng_num]+' '
    return phone_num

def to_bn_word(number):
    """
    Takes a number and outputs the word form in Bengali for that number.
    """
    unedited_text = number
    generated_words = ""
    maybe_year = False
    is_phone_number = True

    if len(number) > 10 or (number[0] == '০' and '.' not in number):
        return parse_phone_or_id_number(number)

    if len(number) == 4 and '.' not in number:
        maybe_year = True
    number = input_sanitizer(number)

    if number is not None:
        if maybe_year:
            first_half = math.floor(number/100)
            second_half = number%100

            if first_half < 20:
                generated_words += numeric_words[str(first_half)] + " শত " + numeric_words[str(second_half)]
                return generated_words

        whole, fraction = float_int_extraction(number)

        whole_segments = generate_segments(whole)

        generated_words = whole_part_word_gen(whole_segments)

        if fraction:
            if generated_words:
                return generated_words + " দশমিক " + fraction_to_words(fraction)
            else:
                return "দশমিক " + fraction_to_words(fraction)
        else:
            return generated_words
    
    else:
        return unedited_text


def replace_number_with_text(text):
    idx = 0
    new_text = text
    while idx < len(text):
        if text[idx] in numbers:
            num = ""
            start = idx
            while text[idx] in numbers:
                num += text[idx]
                if text[idx+1] == ',':
                    idx += 1
                if text[idx+1] == '.':
                    num += '.'
                    idx += 1
                idx += 1
    
            end = idx
            
            num_to_bn_word = to_bn_word(num)
            segment1 = text[0:start]
            segment2 = text[end:]


            text = segment1 + num_to_bn_word + segment2
            new_text = text
        idx += 1

    return new_text


attribution_dict = {
    "সাঃ": "সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম",
    "আঃ": "আলাইহিস সালাম",
    "রাঃ": "রাদিআল্লাহু আনহু",
    "রহঃ": "রহমাতুল্লাহি আলাইহি",
    "রহিঃ": "রহিমাহুল্লাহ",
    "হাফিঃ": "হাফিযাহুল্লাহ",
    "বায়ান": "বাইআন",
    "দাঃবাঃ": "দামাত বারাকাতুহুম,দামাত বারাকাতুল্লাহ",
    # "আয়াত" : "আইআত",#আইআত
    # "ওয়া" : "ওআ",
    # "ওয়াসাল্লাম"  : "ওআসাল্লাম",
    # "কেন"  : "কেনো",
    # "কোন" : "কোনো",
    # "বল"   : "বলো",
    # "চল"   : "চলো",
    # "কর"   : "করো",
    # "রাখ"   : "রাখো",
    "’": "",
    "‘": "",
    # "য়"     : "অ",
    # "সম্প্রদায়" : "সম্প্রদাই",
    # "রয়েছে"   : "রইছে",
    # "রয়েছ"    : "রইছ",
    "/": " বাই ",
}


def tag_text(text: str):
    # remove multiple spaces
    text = re.sub(" +", " ", text)
    # create start and end
    text = "start" + text + "end"
    # tag text
    parts = re.split("[\u0600-\u06FF]+", text)
    # remove non chars
    parts = [p for p in parts if p.strip()]
    # unique parts
    parts = set(parts)
    # tag the text
    for m in parts:
        if len(m.strip()) > 1:
            text = text.replace(m, f"{m}")
    # clean-tags
    text = text.replace("start", "")
    text = text.replace("end", "")
    return text


def normalize(sen):
    global bnorm  # pylint: disable=global-statement
    _words = [bnorm(word)["normalized"] for word in sen.split()]
    return " ".join([word for word in _words if word is not None])


def expand_full_attribution(text):
    for word, attr in attribution_dict.items():
        if word in text:
            text = text.replace(word, normalize(attr))
    return text


def collapse_whitespace(text):
    # Regular expression matching whitespace:
    _whitespace_re = re.compile(r"\s+")
    return re.sub(_whitespace_re, " ", text)


def bangla_text_normalize(text: str) -> str:
    # english numbers to bangla conversion
    res = re.search("[0-9]", text)
    if res is not None:
        text = bangla.convert_english_digit_to_bangla_digit(text)

    # replace ':' in between two bangla numbers with ' এর '
    pattern = r"[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]:[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]"
    matches = re.findall(pattern, text)
    for m in matches:
        r = m.replace(":", " এর ")
        text = text.replace(m, r)

    # numerize text
    # text = numerize(text)

    # tag sections
    text = tag_text(text)

    # text blocks
    # blocks = text.split("")
    # blocks = [b for b in blocks if b.strip()]

    # create tuple of (lang,text)
    if "" in text:
        text = text.replace("", "").replace("", "")
    # Split based on sentence ending Characters
    bn_text = text.strip()

    sentenceEnders = re.compile("[।!?]")
    sentences = sentenceEnders.split(str(bn_text))

    data = ""
    for sent in sentences:
        res = re.sub("\n", "", sent)
        res = normalize(res)
        # expand attributes
        res = expand_full_attribution(res)

        res = collapse_whitespace(res)
        data += res
    return data
