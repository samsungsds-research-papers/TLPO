"""
The MIT License

Copyright (c) 2026 Samsung SDS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import re
import unicodedata as ud
import jieba

from fugashi import Tagger
from dataclasses import dataclass
from lingua import Language, LanguageDetectorBuilder
from emoji import is_emoji

@dataclass
class ConsistencyResult:
    response_failure_rate: float = -1
    response_pass: int = -1
    response_fail: int = -1
    line_failure_rate: float = -1
    line_pass: int = -1
    line_fail: int = -1
    word_failure_rate: float = -1
    word_pass: int = -1
    word_fail: int = -1


zh_word_split = jieba
zh_word_split.initialize()
ja_word_split = Tagger("-O wakati -b 50000")

LC_DETECTOR_VERSION = "1.1.2"

LINGUA_LANGUAGES = [Language.ENGLISH,
                    Language.SPANISH,
                    Language.GERMAN,
                    Language.FRENCH,
                    Language.INDONESIAN,
                    Language.ITALIAN,
                    Language.PORTUGUESE,
]

LINGUA_DETECTOR = LanguageDetectorBuilder.from_languages(*LINGUA_LANGUAGES).build()

LINGUA_CODE = {
    "ENGLISH" : "en",
    "SPANISH" : "es",
    "GERMAN" : "de",
    "FRENCH" : "fr",
    "INDONESIAN" : "id",
    "ITALIAN" : "it",
    "PORTUGUESE" : "pt",
}

LINGUA_THRESHOLD = 0.1

LATIN_SCRIPT_LANGUAGES = ["en","es","de","fr","id","it","pt"]


EXCEPTION_CODE_NAME_LIST = [
    "MATHEMATICAL","MODIFIER","DOUBLE-STRUCK","SUPERSCRIPT","PLANK","MICRO","EULER","OHM","KELVIN","ANGSTROM",
    "SUBSCRIPT","GREEK SMALL LETTER","GREEK CAPITAL LETTER","LATIN SUBSCRIPT SMALL LETTER"
]


TARGET_CODE_NAME_LIST_DICT = {
    "ja" : ["HIRAGANA","KATAKANA","KATAKANA-HIRAGANA","CJK","HALFWIDTH KATAKANA",
            "HALFWIDTH KATAKANA-HIRAGANA","IDEOGRAPHIC ITERATION MARK"],
    "ko" : ["HANGUL"],
    "zh" : ["CJK"],
    "ar" : ["ARABIC"],
    "hi" : ["DEVANAGARI"],
    "ru" : ["CYRILLIC"],
}


EXCEPTION_CHARACTER_SET = {
    'ℎ','ϫ','⑦','①','Ͻ','Ϫ','ϕ','ℳ','ϩ','⑭','Ϯ','ϗ','Ϭ','ℑ','ϴ','ϒ','ϯ','ⅎ','ℜ','⑮','Ϧ','Ϡ','ϳ','⑥','ℴ','ϼ','ℬ','ℓ','ℌ',
    'ϱ','ℐ','ℏ','ℋ','Ϗ','ϖ','ℷ','ϲ','ℰ','ϧ','ℨ','ϥ','ℵ','Ͼ','Ϩ','ϣ','ͻ','Ϙ','ℶ','ℛ','Ϟ','⑫','Ϥ','⑤','Ϛ','ϰ','ϓ','⑨',
    'Ͽ','Ϣ','ϵ','ℸ','ℹ','Ϲ','ℊ','⑧','ϐ','④','ͽ','ℱ','ʹ','⑪','②','ϑ','Å','Ⅎ','ℒ','⑩','Ϝ','ϔ','ͼ','ℯ','ℭ','ϭ','⑬','③','ͺ',
    'ø',
}

CHINESE_TONE_SET = {
    'ā','ō','ē','ī','ū','ǖ', # lowercase first tone
    'á','ó','é','í','ú','ǘ', # lowercase second tone
    'ǎ','ǒ','ě','ǐ','ǔ','ǚ', # lowercase third tone
    'à','ò','è','ì','ù','ǜ', # lowercase fourth tone
    'Ā','Ō','Ē','Ī','Ū','Ǖ', # uppercase first tone
    'Á','Ó','É','Í','Ú','Ǘ', # uppercase second tone
    'Ǎ','Ǒ','Ě','Ǐ','Ǔ','Ǚ', # uppercase third tone
    'À','Ò','È','Ì','Ù','Ǜ', # uppercase fourth tone
}

EXCEPTION_MULTICHARACTER_SET = {
    'inhg','pa','hz','gal','μg','kn','ms','h','db','s','μm','exp','kgf','f','ft','ml','oz','μs','mg','lx','nm','tb',
    'bbl','in','d','mb','tan','det','lm','mmhg','p','ghz','qt','gy','acre','cm','lbf','hv','a','u','w','oe','da','atm',
    'm','c','khz','erg','gb','sv','mw','kl','dl','kt','n','mhz','kb','t','°f','fl','ev','hp','au','km','pt','mi','dyn',
    'ha','kat','l','btu','mx','g','min','bar','ly','j','mol','sq','kg','°c','sin','torr','ln','st','lim','bq','yd','v',
    'psi','mm','k','kwh','kw','cd','cos','log','pc','wb','ppm', 'byte', 'bit', 'mph', 'x', 
    'json', 'html', 'css', 'python', 'c++', 'java', 'vcpu', 'cpu', 'ram', 'https'
}

phonetic_range = (
    r'\u0250-\u02AF'  # IPA Extensions
    r'\u1D00-\u1D7F'  # Phonetic Extensions
    r'\u1D80-\u1DBF'  # Phonetic Extensions Supplement
    r'\u02B0-\u02FF'  # Spacing Modifier Letters
    r'\u0300-\u036F'  # Combining Diacritical Marks
)

PHONETIC_PATTERN = f'[{phonetic_range}]'

EXCEPTION_UNICODE_BLOCKS = [
    # (int("0x2190", 16), int("0x21FF", 16)), # Arrows, 112
    # (int("0x2200", 16), int("0x22FF", 16)), # Mathematical Operators, 256
    # (int("0x27C0", 16), int("0x27EF", 16)), # Miscellaneous Mathematical Symbols-A, 48
    # (int("0x2980", 16), int("0x29FF", 16)), # Miscellaneous Mathematical Symbols-B, 128
    (int("0x1D400", 16), int("0x1D7FF", 16)), # Mathematical Alphanumeric Symbols, 1024
    # (int("0x20A0", 16), int("0x20CF", 16)), # Currency Symbols, 48
    (int("0x2150", 16), int("0x218F", 16)), # Number Forms, 64
    (int("0x2100", 16), int("0x214F", 16)), # Letterlike Symbols, 80
]


def _is_ascii(character: str) -> bool:
    ascii_value  = ord(character)

    if ascii_value < 0 or ascii_value > 127:    # not ascii
        return False

    return True

    
def _is_ascii_and_alphabet(character: str) -> bool:
    ascii_value  = ord(character)

    if ascii_value < 0 or ascii_value > 127:    # not ascii
        return False

    if (ascii_value >= 65 and ascii_value <= 90) or (ascii_value >= 97 and ascii_value <= 122): # alphabetic
        return True

    return False


def _is_ascii_and_not_alphabet(character: str) -> bool:
    ascii_value  = ord(character)

    if ascii_value < 0 or ascii_value > 127:    # not ascii
        return False

    if (ascii_value >= 65 and ascii_value <= 90) or (ascii_value >= 97 and ascii_value <= 122): # alphabetic
        return False

    return True


def _is_ascii_uppercase_alphabet(character: str) -> bool:
    ascii_value  = ord(character)

    if ascii_value < 0 or ascii_value > 127:    # not ascii
        return False

    if ascii_value >= 65 and ascii_value <= 90: # uppercase alphabetic
        return True

    return False


def _starts_uppercase_alphabet(word: str) -> bool:

    first_letter = word[0]
    
    try:
        if word == "'" or word == "\t" or word == "	":
            return False

        if "LATIN CAPITAL" in ud.name(first_letter):
            return True
    
        first_letter_after_apostrophe = word.split("'")[-1][0] # Splitting by "'" for Romance Languages (d'Arc, etc..)
    
        if "LATIN CAPITAL" in ud.name(first_letter_after_apostrophe):
            return True
    except Exception as e:
        print("[_starts_uppercase_alphabet]: {}, {}".format(word, e))
        return False
    
    return False


def _is_codename_in_list(code_name:int, CODE_LIST:list) -> bool:
    for code in CODE_LIST:
        if code in code_name:
            return True
        
    return False


def is_exception_unicode_blocks(character: str) -> bool:
    unicode_code_point = ord(character)

    for start_point, end_point in EXCEPTION_UNICODE_BLOCKS:
        if start_point <= unicode_code_point and unicode_code_point < end_point:
            return True


def is_special_character_unicode(character: str):
    # Po: general punctuation (Punctuation, other)
    # Ps: opening punctuation (Punctuation, open)
    # Pe: closing punctuation (Punctuation, close)
    # Sm: mathematical symbol (Symbol, math)
    # Sk: modifier symbol (Symbol, modifier)
    # So: other symbol (Symbol, other)
    # Cc: control character (Control)

    try:
        category = ud.category(character)
        # Treat punctuation, symbols, and control characters as special characters
        if category.startswith('P') or category.startswith('S') or category.startswith('C'):
            return True
    except Exception as e:
        print("[is_special_character_unicode] unknown category: {}, {}".format(character, e))
        return False

    return False


def is_chinese_tone_character(character:str, target_language:str):
    return target_language == "zh" and character in CHINESE_TONE_SET


def _check_char(character:str, target_language:str, ignore_english:bool):

    if is_emoji(character):
        return 0, 0
    
    if is_special_character_unicode(character):
        return 0, 0
    
    if character.isspace():
        return 0, 0

    if ignore_english:
        # ignore all ascii codes 
        if _is_ascii(character):
            return 0, 0 # pass=0, fail=0
    else:
        # ignore ascii codes excluding the alphabet
        if _is_ascii_and_not_alphabet(character):
            return 0, 0 # pass=0, fail=0

    if is_exception_unicode_blocks(character):
        return 0, 0 # pass=0, fail=0

    if is_chinese_tone_character(character, target_language):
        return 0, 0 # pass=0, fail=0

    try:
        code_name = ud.name(character)
    except Exception as e:
        print("[_check_char] ud.name({}) An error occurred: {}".format(character, e))
        return 0, 0 # pass=0, fail=0

    # Check UNICODE name
    if _is_codename_in_list(code_name, EXCEPTION_CODE_NAME_LIST):
        return 0, 0 # pass=0, fail=0
    
    # Check UNICODE name
    TARGET_CODE_NAME_LIST = TARGET_CODE_NAME_LIST_DICT[target_language]

    if _is_codename_in_list(code_name, TARGET_CODE_NAME_LIST):
        return 1, 0 # pass=1, fail=0

    return 0, 1 # pass=0, fail=1, Language Confusion as a non-special/number character detected


def _get_letter_length(word:str) -> int:
    letter_count = 0

    for character in word:
        try:
            if _is_ascii_and_not_alphabet(character):
                continue
            elif _is_codename_in_list(ud.name(character), EXCEPTION_CODE_NAME_LIST):
                continue

            letter_count += 1
        except Exception as e:
            print("[_get_letter_length] ud.name({}) An error occurred: {}".format(character, e))
            continue
    
    return letter_count


def _check_word(word:str, target_language:str, ignore_english):
    if len(word) == 0:
        return 0, 0, 0
    
    if _starts_uppercase_alphabet(word):    # ignore english chars starting with uppercase alphabet
        ignore_english = True
        
    if target_language in LATIN_SCRIPT_LANGUAGES:
        return _check_word_latin(word, target_language, ignore_english)
    else:
        return _check_word_not_latin(word, target_language, ignore_english)


def _check_word_latin(word:str, target_language: str, ignore_english):
    word_pass = 0
    word_fail = 0

    if _get_letter_length(word) <= 1:
        return 0, 0, 0 # Skip if the length excluding numbers and special characters is 1 or less
    
    confidence_values = LINGUA_DETECTOR.compute_language_confidence_values(word)

    for conf in confidence_values:
        language = LINGUA_CODE[conf.language.name]
        probability = conf.value
        
        if probability > LINGUA_THRESHOLD and language == target_language:
            return 1, 0, 0 # pass=1, fail=0


    if ignore_english==True:
        for conf in confidence_values:
            language = LINGUA_CODE[conf.language.name]
            probability = conf.value
            
            if probability > LINGUA_THRESHOLD and language == 'en':
                return 0, 0, 0 # pass=0, fail=0

             
    return 0, 1, 0 # pass=0, fail=1


def _check_word_not_latin(word:str, target_language:str, ignore_english:bool):
    word_pass = 0
    word_fail = 0
    
    char_pass_total = 0
    char_fail_total = 0

    fail_offset = None
    
    for idx, character in enumerate(word):
        char_pass, char_fail = _check_char(character, target_language, ignore_english)
        char_pass_total += char_pass
        char_fail_total += char_fail

        if fail_offset==None and char_fail>0:
            fail_offset = idx
        
    if char_fail_total > 0:
        word_pass = 0
        word_fail = 1
    elif char_pass_total > 0:
        word_pass = 1
        word_fail = 0

    return word_pass, word_fail, fail_offset


def _get_valid_word_count(words:list, target_language:str, ignore_english):
    words_count = 0
    
    for word in words:
        word_pass, word_fail, _ = _check_word(word, target_language, ignore_english)
        words_count += word_pass
        words_count += word_fail

    return words_count


def _split_line_into_words(line:str, lang:str) -> list[str]:
    global ja_word_split, zh_word_split

    # remove latex symbol patterns
    latex_pattern = r'\\[a-zA-Z]+'
    line_cleaned = re.sub(latex_pattern, ' ', line)
    
    if lang == 'zh':
        return list(zh_word_split.cut(line_cleaned))
        
    elif lang == 'ja':
        return ja_word_split.parse(line_cleaned).split()

    return re.findall(r"\b[\w'-]+\b", line_cleaned, flags=re.UNICODE)



def _check_line(line:str, target_language:str, ignore_english):
    line_pass = 0      # line pass
    line_fail = 0

    words = _split_line_into_words(line, target_language)        

    if _get_valid_word_count(words, target_language, False) < 5:
        return 0, 0

    word_pass_total = 0
    word_fail_total = 0
    
    for word in words:        
        word_pass, word_fail, _ = _check_word(word, target_language, ignore_english)
    
        word_pass_total += word_pass
        word_fail_total += word_fail

    word_total = word_pass_total+word_fail_total
    if word_total==0:
        return line_pass, line_fail

    line_pass = 1 if word_pass_total >= word_fail_total else 0
    line_fail = 1 - line_pass

    return line_pass, line_fail
    

def exclude_words(text, exclude_list):
    """
    Removes words from exclude_list in the string based on conditions, ignoring case.
    - Removes only if there are digits, spaces, or non-Latin characters before/after.
    - Does not remove if Latin letters (a-zA-Z) are directly attached before/after.
    """
    patterns = []
    for word in exclude_list:
        # Pattern: No Latin letter before, preceded by digit/space/non-Latin or start,
        # followed by digit/space/non-Latin or end, and no Latin letter after.
        pattern = r'(?<![a-zA-Z])(?:(?<=\d)|(?<=\s)|(?<=[^a-zA-Z0-9])|^)' + re.escape(word) + r'(?=[\d\s]|[^a-zA-Z0-9]|$)(?![a-zA-Z])'
        patterns.append(pattern)
    
    combined_pattern = '|'.join(patterns)
    
    def replacer(match):
        return ''
    
    # Use re.IGNORECASE to make matching case-insensitive
    return re.sub(combined_pattern, replacer, text, flags=re.IGNORECASE)


def exclude_email_url(text):
    """
    Removes email addresses and URLs from the string.
    - Matches and removes patterns like 'example@email.com' or 'https://www.example.com'.
    """
    # Email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # URL pattern (including http, https, www)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}[^\s]*'
    
    combined_pattern = f'({email_pattern})|({url_pattern})'
    
    def replacer(match):
        return ''
    
    # Case-insensitive matching (emails/URLs often mixed case)
    return re.sub(combined_pattern, replacer, text, flags=re.IGNORECASE)


def _remove_exceptions(text:str):
    for exception in EXCEPTION_CHARACTER_SET:
        text = text.replace(exception, '')

    text = exclude_words(text, EXCEPTION_MULTICHARACTER_SET)
    text = exclude_email_url(text)
    text = re.sub(PHONETIC_PATTERN, '', text)
    
    return text        
        
                
def check_response(response:str, target_language:str, ignore_english:bool=True):

    line_pass_total = 0      # line pass
    line_fail_total = 0
    word_pass_total = 0      # word pass
    word_fail_total = 0
    resp_pass = 0
    resp_fail = 0

    response = _remove_exceptions(response)

    lines = response.split("\n")
    
    for line in lines: # Split by line
        if len(line) == 0:
            continue

        line_pass, line_fail = _check_line(line, target_language, ignore_english)
        line_pass_total += line_pass
        line_fail_total += line_fail
        
        #if line_pass == 1:        
        words = _split_line_into_words(line, target_language)                    
                
        for word in words:
            word_pass, word_fail, _ = _check_word(word, target_language, ignore_english)
            word_pass_total += word_pass
            word_fail_total += word_fail

        
    if line_pass_total + line_fail_total + word_pass_total + word_fail_total > 0:
        if line_fail_total + word_fail_total > 0:
            resp_fail = 1
        else:
            resp_pass = 1
            

    return resp_pass, resp_fail, line_pass_total, line_fail_total, word_pass_total, word_fail_total


def get_all_response_consistency(all_response:list, target_language: str, ignore_english:bool=True):
    word_pass = 0
    word_fail = 0

    line_pass = 0
    line_fail = 0
    
    resp_pass = 0
    resp_fail = 0
    
    for response in all_response:
        resp_p, resp_f, line_p, line_f, word_p, word_f = check_response(response, target_language, ignore_english)

        resp_pass += resp_p
        resp_fail += resp_f

        line_pass += line_p
        line_fail += line_f
        
        word_pass += word_p
        word_fail += word_f            

    result = ConsistencyResult()

    result.response_pass = resp_pass
    result.response_fail = resp_fail
    result.response_failure_rate = resp_fail * 100 / (resp_pass + resp_fail) if (resp_pass + resp_fail)>0 else 0

    result.line_pass = line_pass
    result.line_fail = line_fail
    result.line_failure_rate = line_fail * 100 / (line_pass + line_fail) if (line_pass + line_fail)>0 else 0

    result.word_pass = word_pass
    result.word_fail = word_fail
    result.word_failure_rate = word_fail * 100 / (word_pass + word_fail) if (word_pass + word_fail)>0 else 0

    return result


def get_confusion_point(response:str, target_language:str, ignore_english:bool=True):

    response_cleaned = _remove_exceptions(response)
    confusion_word = None
    confusion_offset = 0
    
    lines = response_cleaned.split("\n")
    
    for line in lines: # Split by line
        if len(line) == 0:
            continue
        
        words = _split_line_into_words(line, target_language)                                    
        for word in words:
            word_pass, word_fail, confusion_offset = _check_word(word, target_language, ignore_english)

            if word_fail > 0:
                confusion_word = word
                break

        if confusion_word != None:
            break

    cp = -1                 
    if confusion_word != None:
        cp_word_start = response.find(confusion_word)

        if target_language in LATIN_SCRIPT_LANGUAGES:
            cp = cp_word_start
        else:        
            for check_pos in range(cp_word_start+confusion_offset, len(response), 1):
                char_pass, char_fail = _check_char(response[check_pos], target_language, ignore_english)
                if char_fail>0:
                    cp = check_pos
                    break        
        
    return cp
    


