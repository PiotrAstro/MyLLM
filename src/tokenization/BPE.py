import regex
import os
import pathlib
import numpy as np
import dataclasses

# I am sitting here and thinking:
# Why do people use python? it is embarassingly slow.
# After trying Julia it is hard to go back to python and just wait 100x longer for the same result.
# Unfortunately people usually don't know julia so it is better to do it in python.
# Maybe this text will convince somone to take a look at Julia :)
# What am I doing with my life that I am writing this comment?
# I should be doing something more productive.

@dataclasses.dataclass
class _ConstructWord:
    word: str
    tokens: list[int]
    pairs: list[tuple[int, int]]
    counter: int

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, _ConstructWord):
            return False
        return self.word == value.word
    
    def __hash__(self) -> int:
        return hash(self.word)

class BPE_Tokenizer:
    encoding: str = "utf-8"
    special_tokens: list[str]
    special_tokens_reverse: dict[str, int]
    special_tokens_split: regex.Pattern[str]
    bytes_to_unicode_list: list[str]
    unicode_to_bytes: dict[str, int]
    split_formula: regex.Pattern[str]
    tokens_mapping: list[bytearray] = []
    tokens_merge_ranking: dict[tuple[int, int], int]  # key: merged tokens, value: new token id, it is also ranking itself
    cache: dict[str, list[int]]

    def __init__(self, special_tokens: None | list[str] = None):
        """
        It is only for internal use
        """
        self.special_tokens = special_tokens if special_tokens else self.construct_special_tokens()
        self.special_tokens_reverse = {token: position for (position, token) in enumerate(self.special_tokens)}
        self.special_tokens_split = regex.compile(f"({'|'.join(regex.escape(token) for token in self.special_tokens)})")
        self.bytes_to_unicode_list = self.bytes_to_unicode()
        self.unicode_to_bytes = {key: value for (value, key) in enumerate(self.bytes_to_unicode_list)}
        self.split_formula = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def decode(self, tokens: list[int] | np.ndarray, error: str = "replace") -> str:
        """
        Decoding when object was previously initialized
        One can decide what to do with errors.
        """
        assert self.tokens_mapping  # check if it initialized or loaded

        bytearrays = [
            self.tokens_mapping[token_id]
            if token_id < len(self.tokens_mapping)
            else bytearray(self.special_tokens[token_id - len(self.tokens_mapping)], encoding=self.encoding)
            for token_id in tokens
        ]
        final_bytes = bytearray(b'').join(bytearrays)
        final_text = final_bytes.decode(self.encoding, errors=error)
        return final_text
    
    def load_and_encode(self, path: pathlib.Path, file_text_delimiter: str = "<|endoftext|>") -> np.ndarray:
        """
        Loading and encoding when object was not initialized
        """
        arrays = []
        for text in self._load_texts(path):
            arrays.append(self.encode(text))
            arrays.append(self.encode(file_text_delimiter))
        return np.concatenate(arrays)
        
    def encode(self, text: str) -> np.ndarray:
        """
        Encoding when object was previously initialized
        """
        assert self.tokens_mapping  # check if it initialized or loaded
        final_encoded = []
        for text_part in regex.split(self.special_tokens_split, text):
            if text_part in self.special_tokens:
                final_encoded.append(self.special_tokens_reverse[text_part] + len(self.tokens_mapping)) 
            else:
                splitted = regex.findall(self.split_formula, text_part)
                for word in splitted:
                    final_encoded.extend(self._word_to_tokens(word))
        return np.array(final_encoded, dtype=np.int32)

    def is_special_token(self, token: int, special_token: str) -> bool:
        if token >= len(self.tokens_mapping):
            return self.special_tokens[token - len(self.tokens_mapping)] == special_token
        else:
            return False
    
    def get_tokens_number(self) -> int:
        return len(self.tokens_mapping) + len(self.special_tokens)

    def construct(self, path: pathlib.Path, token_normal_number: int):
        words: dict[str, _ConstructWord] = {}  # word str, list of tokens, number of occurances
        for text_part in self._load_texts(path):
            for text_part_splitted in regex.split(self.special_tokens_split, text_part):
                if text_part_splitted not in self.special_tokens:
                    for word in regex.findall(self.split_formula, text_part_splitted):
                        if word in words.keys():
                            words[word].counter += 1
                        else:
                            word_tokens = [b for b in bytearray(word, self.encoding)]
                            if len(word_tokens) > 1:
                                pairs = [(word_tokens[i], word_tokens[i + 1]) for i in range(len(word_tokens) - 1)]
                                words[word] = _ConstructWord(
                                    word=word,
                                    tokens=word_tokens,
                                    pairs=pairs,
                                    counter=1,
                                )
        self._contruct_word_tokens([w for w in words.values()], token_normal_number)

    def save(self, path: pathlib.Path):
        assert self.tokens_mapping  # check if it initialized or loaded

        reversed_ranking_dict = {position: merge for (merge, position) in self.tokens_merge_ranking.items()}
        text_list = []
        for (position, token_map) in enumerate(self.tokens_mapping):
            if position in reversed_ranking_dict.keys():
                pos1, pos2 = reversed_ranking_dict[position]
                text_list.append(" ".join([self._bytes_to_unicode(token_map), str(pos1), str(pos2)]))
            else:
                text_list.append(self._bytes_to_unicode(token_map))

        with open(path, "w", encoding=self.encoding) as f:
            f.writelines("\n".join(text_list))

    def load(self, path: pathlib.Path):
        self.tokens_mapping = []
        self.tokens_merge_ranking = {}
        with open(path, "r", encoding=self.encoding) as f:
            for line in f.readlines():
                splitted = line.split()
                if len(splitted) == 1:
                    self.tokens_mapping.append(self._unicode_to_bytes(splitted[0]))
                elif len(splitted) == 3:
                    token_text, first_pair_text, second_pair_text = splitted
                    self.tokens_mapping.append(self._unicode_to_bytes(token_text))
                    key = (int(first_pair_text), int(second_pair_text))
                    self.tokens_merge_ranking[key] = len(self.tokens_mapping) - 1
                else:
                    raise IOError("error loading token")
    
    def _load_texts(self, path: pathlib.Path) -> list[str]:
        if os.path.isdir(path):
            documents_to_read = [os.path.join(path, file) for file in os.listdir(path)]
        else:
            documents_to_read = [path]

        text_parts = []  # word str, list of tokens, number of occurances
        for file in documents_to_read:
            with open(file, "r", encoding=self.encoding) as f:
                text = f.read()
            text_parts.append(text)
        return text_parts

    def _unicode_to_bytes(self, unicode_word: str) -> bytearray:
        return bytearray(
            [self.unicode_to_bytes[char] for char in unicode_word]
        )
    
    def _bytes_to_unicode(self, bytes_word: bytearray) -> str:
        return "".join([self.bytes_to_unicode_list[byte] for byte in bytes_word])

    def _word_to_tokens(self, word: str) -> list[int]:
        if word in self.cache.keys():
            return self.cache[word]
        not_valid_token = len(self.tokens_mapping)
        word_tokens = [b for b in bytearray(word, self.encoding)]
        pairs = []
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            ranking = self.tokens_merge_ranking.get(pair, not_valid_token)
            pairs.append((pair, ranking))

        while len(pairs) > 0:
            argmin = min(range(len(pairs)), key=lambda x: pairs[x][1])
            new_token = pairs[argmin][1]
            if new_token == not_valid_token:
                break
            else:
                del word_tokens[argmin]
                word_tokens[argmin] = new_token

                del pairs[argmin]
                argmin_prev = argmin - 1
                argmin_next = argmin

                if argmin_next < len(pairs):
                    pair = (word_tokens[argmin_next], word_tokens[argmin_next + 1])
                    ranking = self.tokens_merge_ranking.get(pair, not_valid_token)
                    pairs[argmin_next] = (pair, ranking)
                if argmin_prev >= 0:
                    pair = (word_tokens[argmin_prev], word_tokens[argmin_prev + 1])
                    ranking = self.tokens_merge_ranking.get(pair, not_valid_token)
                    pairs[argmin_prev] = (pair, ranking)
        self.cache[word] = word_tokens
        return word_tokens
    
    def _contruct_word_tokens(self, word_list: list[_ConstructWord], desired_tokens_n: int):
        tokens_mapping: list[bytearray] = [bytearray(chr(i), encoding=self.encoding) for i in range(2**8)]
        pairs_counter: dict[tuple[int, int], int] = {}
        pairs_word_backbond: dict[tuple[int, int], set[_ConstructWord]] = {}
        tokens_merge_ranking: dict[tuple[int, int], int] = {}

        for word in word_list:
            for pair in word.pairs:
                pairs_counter[pair] = pairs_counter.get(pair, 0) + word.counter
                word_set = pairs_word_backbond.get(pair, set())
                word_set.add(word)
                pairs_word_backbond[pair] = word_set

        while len(tokens_mapping) < desired_tokens_n:
            self._print_progress(len(tokens_mapping), desired_tokens_n)
            max_pair, _ = max(pairs_counter.items(), key=lambda x: x[1])
            new_token_value = len(tokens_mapping)
            tokens_merge_ranking[max_pair] = new_token_value
            tokens_mapping.append(tokens_mapping[max_pair[0]] + tokens_mapping[max_pair[1]])
            
            # Update pairs
            for word in pairs_word_backbond[max_pair]:
                word_tokens = word.tokens
                pairs = word.pairs
                for (i, pair) in enumerate(pairs):
                    if pair == max_pair:
                        del word_tokens[i]
                        word.tokens[i] = new_token_value
                        del pairs[i]
                        for pos in [i - 1, i]:
                            if pos < len(pairs) and pos >= 0:
                                pair = (word_tokens[pos], word_tokens[pos + 1])
                                pairs[pos] = pair
                                pairs_counter[pair] = pairs_counter.get(pair, 0) + word.counter
                                word_set = pairs_word_backbond.get(pair, set())
                                word_set.add(word)
                                pairs_word_backbond[pair] = word_set
            del pairs_counter[max_pair]
            del pairs_word_backbond[max_pair]
        self.tokens_mapping = tokens_mapping
        self.tokens_merge_ranking = tokens_merge_ranking

    def _print_progress(self, len_tokens_mapping: int, desired_tokens_n: int):
        # save cursor
        print("\033[s", end="", flush=True)
        print(f"Constructing tokens: {len_tokens_mapping} / {desired_tokens_n}", end="", flush=True)
        # restore cursor
        print("\033[u", end="", flush=True)

    @staticmethod
    def bytes_to_unicode() -> list[str]:
        """
        This method outputs mapping of bytes (ascii) to printable characters that can be printed
        """

        # initial ascii printable characters
        ascii_printable = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        bytes_printable = []
        byte_size = 2**8
        above_ascii = 0
        for byte in range(byte_size):
            if byte in ascii_printable:
                bytes_printable.append(chr(byte))
            else:
                bytes_printable.append(chr(byte_size + above_ascii))
                above_ascii += 1
        return bytes_printable
    
    @staticmethod
    def construct_special_tokens() -> list[str]:
        """
        This method outputs mapping of bytes (ascii) to printable characters that can be printed
        """

        # special characters with distinct meaning
        return [
            "<|endoftext|>",
        ]

