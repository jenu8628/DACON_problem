from hanspell import spell_checker
from tensorflow.keras.utils import to_categorical

# a = [1, 3, 4, 5, 6, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3]
#
#
# print(to_categorical(a))


a = '별로입니자별로입니다'
spelled_sentence = spell_checker.check(a)
hanspell_sentence = spelled_sentence.checked
print(hanspell_sentence)
