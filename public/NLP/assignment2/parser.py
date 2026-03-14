import csv
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy

ROOT_IDX = -1


class PartialParse:
    def __init__(self, words):
        self.words = words
        self.stack = [ROOT_IDX]
        self.buffer = list(range(len(words)))
        self.dependencies = []

    def shift(self):
        if self.buffer:
            self.stack.append(self.buffer.pop(0))

    def left_arc(self):
        if len(self.stack) >= 2:
            self.dependencies.append((self.stack[-1], self.stack[-2]))
            self.stack.pop(-2)

    def right_arc(self):
        if len(self.stack) >= 2:
            self.dependencies.append((self.stack[-2], self.stack[-1]))
            self.stack.pop(-1)

    @property
    def is_complete(self):
        return not self.buffer and len(self.stack) == 1

    def word(self, idx):
        return "ROOT" if idx == ROOT_IDX else self.words[idx]

    def stack_str(self):
        return str([self.word(i) for i in self.stack])

    def buffer_str(self):
        return str([self.word(i) for i in self.buffer])


class NeuralParserOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_proj = nn.Linear(2, 3)
        self._nlp = spacy.load("en_core_web_sm")
        self._heads = {}

    def build_gold_arcs(self, sentence):
        doc, words = self._nlp(sentence), sentence.split()
        spacy_to_split, char_cursor = {}, 0
        for i, word in enumerate(words):
            start = sentence.index(word, char_cursor)
            char_cursor = start + len(word)
            for tok in doc:
                if start <= tok.idx < char_cursor:
                    spacy_to_split[tok.i] = i
        for tok in doc:
            i = spacy_to_split.get(tok.i)
            h = spacy_to_split.get(tok.head.i)
            if i is not None and h is not None and i not in self._heads:
                self._heads[i] = ROOT_IDX if i == h else h
        return {words[i]: ("ROOT" if h == ROOT_IDX else words[h])
                for i, h in self._heads.items()}

    def predict_transition(self, state):
        if len(state.stack) < 2:
            return "SHIFT"
        if not state.buffer and state.stack[-2] == ROOT_IDX:
            return "RIGHT-ARC"
        s1, s2 = state.stack[-1], state.stack[-2]
        buf = set(state.buffer)
        def pending(hi): return any(self._heads.get(d) == hi for d in buf)
        if self._heads.get(s2) == s1 and not pending(s2):
            return "LEFT-ARC"
        if self._heads.get(s1) == s2 and not pending(s1):
            return "RIGHT-ARC"
        return "SHIFT"

    def forward(self, stack_depth, buffer_depth):
        return F.softmax(self.feature_proj(
            torch.tensor([float(stack_depth), float(buffer_depth)])), dim=0)


def print_tree(state):
    children = defaultdict(list)
    for head, dep in state.dependencies:
        children[head].append(dep)
    for kids in children.values():
        kids.sort()

    def render(idx, prefix, is_last):
        print(prefix + ("└── " if is_last else "├── ") + state.word(idx))
        kids = children.get(idx, [])
        for i, child in enumerate(kids):
            render(child, prefix + ("    " if is_last else "│   "),
                   i == len(kids) - 1)

    print("  ROOT")
    for i, child in enumerate(children.get(ROOT_IDX, [])):
        render(child, "  ", i == len(children[ROOT_IDX]) - 1)


def run_parser(sentence):
    counter = 1
    while os.path.exists(f"parse_output{counter}.csv"):
        counter += 1

    words = sentence.split()
    state = PartialParse(words)
    model = NeuralParserOracle()

    print(f"Parsing: '{sentence}'\n")
    print("Gold arcs (spaCy):")
    for word, head in model.build_gold_arcs(sentence).items():
        print(f"  {head} -> {word}")
    print()

    col1 = col2 = max(len(str(words)) + 20, 40)
    col3, col4 = 24, 22
    header = f"{'Stack':<{col1}} | {'Buffer':<{col2}} | {'New Dependency':<{col3}} | {'Transition':<{col4}}"
    print(header)
    print("-" * len(header))

    rows = [["Stack", "Buffer", "New Dependency", "Transition"]]
    for row in [[state.stack_str(), state.buffer_str(), "", "Initial Configuration"]]:
        rows.append(row)
        print(
            f"{row[0]:<{col1}} | {row[1]:<{col2}} | {row[2]:<{col3}} | {row[3]:<{col4}}")

    step = 0
    while not state.is_complete and step <= 2 * len(words) + 5:
        t = model.predict_transition(state)
        if t == "SHIFT" and not state.buffer:
            t = "RIGHT-ARC"

        dep = ""
        if t == "SHIFT":
            state.shift()
        elif t == "LEFT-ARC":
            dep = f"{state.word(state.stack[-1])} -> {state.word(state.stack[-2])}"
            state.left_arc()
        elif t == "RIGHT-ARC":
            dep = f"{state.word(state.stack[-2])} -> {state.word(state.stack[-1])}"
            state.right_arc()

        row = [state.stack_str(), state.buffer_str(), dep, t]
        rows.append(row)
        print(
            f"{row[0]:<{col1}} | {row[1]:<{col2}} | {row[2]:<{col3}} | {row[3]:<{col4}}")
        step += 1

    print("\nFinal Dependencies:")
    for h, d in state.dependencies:
        print(f"  ({state.word(h)}, {state.word(d)})")

    print("\nDependency Tree:")
    print_tree(state)

    with open(f"parse_output{counter}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nSaved as parse_output{counter}.csv")


if __name__ == "__main__":
    run_parser(
        "I parsed this sentence correctly"
        )
