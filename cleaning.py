import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
from typing import List
nltk.download('stopwords')

def clean_data(text: str, inference_mode=False) -> str:
    """
    It cleans text string by executing the following tasks:

    1. Removes html tags
    2. Removes URLS
    3. Removes punctuation
    4. Removes unicode
    5. Removes numbers
    6. Removes emojis
    7  Removes abbreviated pronouns
    8. Converts string to lowercase
    9. Deletes common english words (stopwords)
    10. Steams the text

    Args:
        text [str]: text to be processed.
    
    Returns:
        [str]: processed text.
    """

    stopwords_return_list = False if inference_mode else True
    steam_input_list = False if inference_mode else True

    text = remove_html_tag(text)
    text = remove_url(text)
    text = removeUnicode(text)
    text = remove_emojis(text)
    text = remove_numbers(text)
    text = to_lower(text)
    text = transform_abbreviated_pronouns(text)
    text = remove_puncutation(text)
    text = stop_words(text, return_list=stopwords_return_list)
    text = stem_text(text, input_list=steam_input_list)
    return text


def stop_words(text: str, return_list=True) -> List[str]:
    """
    It deletes common english (stopwords) from a string. 
    Examples of common words: the, a, an, in, etc.

    Args:
        text [str]
        return_list [bool]: if True  --> the function returns a list.
                            if False --> the function returns a string.

    Returns:
        1. If return_list = True  --> returns text [List[str]]: list of processed text with words not included in the stopwords.
        2. If return_list = False --> returns text [str]: string of processed text with words not included in the stopwords.
    """
    
    stopword = set(stopwords.words('english'))
    text = [word for word in text.split(' ') if word not in stopword]

    if return_list:
        return text
    else:
        return " ".join(text)


def stem_text(text, input_list=True) -> str:
    """
    It stems words from strings in a list using the Snowball Stemmer.

    Args:
        text: list or string containing words (strings) to be stemmed.
        input_list [bool]: if True  --> text is a list.
                           if False --> text is a string.
    
    Returns:
        text [str]: string after applying the stemming algorithm.
    """
    
    ss = SnowballStemmer('english')

    if not input_list:
        text = text.split(' ')

    text = [ss.stem(word) for word in text]
    text = " ".join(text)
    return text


def remove_html_tag(text: str) -> List[str]:
    """
    Remove html tags from text.
    
    Args:
        text [str]: text in which html tags are removed from.
    
    Returns:
        text [str]: text with html tags removed.
    """

    html_tag = re.compile(r'<.*?>')
    text = re.sub(html_tag, '', text)
    return text


def remove_url(text: str) -> str:
    """
    Remove urls present in a string.

    Args:
        text [str]: text in which urls are removed from.
    
    Returns:
        [str]: text with urls removed.
    """

    url = re.compile(r'https://\S+|www\.\S+')
    return re.sub(url, '', text)


def remove_puncutation(text: str) -> str:
    """
    Remove punctionation from a string.
    Examples of punctuation: !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    
    Args:
        text [str]: text in which punctuations are removed from.
    
    Returns:
        [str]: text with punctuation removed.
    """

    return re.sub(r'[^\w\s]', r' ', text)


def remove_numbers(text: str) -> str:
    """
    Removes numbers from a string.
    
    Args:
        text [str]: text in which numbers are removed from.
    
    Returns:
        [str]: text with numbers removed.
    """

    return re.sub(r'[0-9]', '', text)

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r' ', text)       
    text = re.sub(r'[^\x00-\x7f]',r' ',text)
    return text


def remove_emojis(text: str) -> str:
    """
    Remove emojis from a string.

    Args:
        text [str]: text in which emojis are removed from.
    
    Returns:
        [str]: text with emojis removed.
    """

    emoj = re.compile("["
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)


def to_lower(text: str) -> str:
    """
    Convert text to lowercase.
    Args:
        text [str]: text to be converted to lowercase.
    
    Returns:
        [str]: text in lowercase.
    """
    return text.lower()


def transform_abbreviated_pronouns(text: str) -> str:
    """
    Replace abbreviated pronouns to full form expressions from a string.
    Example: 
        don't -> do not
        you're -> you are 
    Args:
        text [str]: text 
    
    Returns:
        [str]: text with abbreviations converted to full form.
    """

    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"We're", "We are", text)
    text = re.sub(r"That's", "That is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"Can't", "Cannot", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"don\x89Ûªt", "do not", text)
    text= re.sub(r"aren't", "are not", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"What's", "What is", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"There's", "There is", text)
    text = re.sub(r"He's", "He is", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"You're", "You are", text)
    text = re.sub(r"I'M", "I am", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"i'm", "I am", text)
    text = re.sub(r"I\x89Ûªm", "I am", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"Isn't", "is not", text)
    text = re.sub(r"Here's", "Here is", text)
    text = re.sub(r"you've", "you have", text)
    text = re.sub(r"you\x89Ûªve", "you have", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"it\x89Ûªs", "it is", text)
    text = re.sub(r"doesn\x89Ûªt", "does not", text)
    text = re.sub(r"It\x89Ûªs", "It is", text)
    text = re.sub(r"Here\x89Ûªs", "Here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"I\x89Ûªve", "I have", text)
    text = re.sub(r"y'all", "you all", text)
    text = re.sub(r"can\x89Ûªt", "cannot", text)
    text = re.sub(r"would've", "would have", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"You\x89Ûªre", "You are", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"Don\x89Ûªt", "Do not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"They're", "They are", text)
    text = re.sub(r"Can\x89Ûªt", "Cannot", text)
    text = re.sub(r"you\x89Ûªll", "you will", text)
    text = re.sub(r"I\x89Ûªd", "I would", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"you're", "you are", text)
    text = re.sub(r"i've", "I have", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"i'll", "I will", text)
    text = re.sub(r"doesn't", "does not",text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"I've", "I have", text)
    text = re.sub(r"Don't", "do not", text)
    text = re.sub(r"I'll", "I will", text)
    text = re.sub(r"I'd", "I would", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"youve", "you have", text)  
    text = re.sub(r"donå«t", "do not", text)  

    return text
