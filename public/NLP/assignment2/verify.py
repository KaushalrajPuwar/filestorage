from q1a_parser import PartialParse, NeuralParserOracle, ROOT_IDX
from collections import defaultdict

# ── Test suite ────────────────────────────────────────────────────────────────
# Each entry: (sentence, gold_deps_or_None, description)
# gold_deps = list of (head_word, dep_word) tuples — None = structural check only
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [

    # ── 1. Assignment sentence ────────────────────────────────────────────────
    (
        "I parsed this sentence correctly",
        [("parsed","I"),("sentence","this"),("parsed","sentence"),
         ("parsed","correctly"),("ROOT","parsed")],
        "Basic: assignment sentence"
    ),

    # ── 2. Simple SVO ─────────────────────────────────────────────────────────
    (
        "She loves him",
        [("loves","She"),("loves","him"),("ROOT","loves")],
        "Basic: simple SVO"
    ),

    # ── 3. Determiner + noun ──────────────────────────────────────────────────
    (
        "The cat sat on the mat",
        [("cat","The"),("sat","cat"),("mat","the"),("on","mat"),
         ("sat","on"),("ROOT","sat")],
        "Basic: det + PP"
    ),

    # ── 4. Negation ───────────────────────────────────────────────────────────
    (
        "I am not a fortunate senator",
        [("am","I"),("am","not"),("senator","a"),("senator","fortunate"),
         ("am","senator"),("ROOT","am")],
        "Negation: copula with negative"
    ),

    # ── 5. Long PP chain ─────────────────────────────────────────────────────
    (
        "All the waste in a year from a nuclear power plant can be stored under a desk",
        [("waste","All"),("waste","the"),("year","a"),("in","year"),
         ("waste","in"),("plant","a"),("plant","nuclear"),("plant","power"),
         ("from","plant"),("waste","from"),("stored","can"),("stored","be"),
         ("stored","waste"),("desk","a"),("under","desk"),("stored","under"),
         ("ROOT","stored")],
        "Long: passive + multiple PPs + duplicate 'a'"
    ),

    # ── 6. Relative clause ────────────────────────────────────────────────────
    (
        "Look what they did to my boy",
        [("did","they"),("did","what"),("boy","my"),("to","boy"),
         ("did","to"),("Look","did"),("ROOT","Look")],
        "Relative clause: wh-complement"
    ),

    # ── 7. Contraction: I'm / can't ───────────────────────────────────────────
    (
        "I'm gonna make him an offer he can't refuse",
        None,
        "Contraction: I'm and can't (structural check)"
    ),

    # ── 8. Contraction: ain't + duplicate word ────────────────────────────────
    (
        "I ain't no loser. I always win",
        None,
        "Contraction + duplicate 'I' (structural check)"
    ),

    # ── 9. Duplicate subject word ─────────────────────────────────────────────
    (
        "I know what I want",
        [("know","I"),("want","what"),("want","I"),("know","want"),
         ("ROOT","know")],
        "Duplicate: 'I' appears twice"
    ),

    # ── 10. Passive voice ─────────────────────────────────────────────────────
    (
        "The book was written by the author",
        [("book","The"),("written","book"),("written","was"),
         ("author","the"),("by","author"),("written","by"),("ROOT","written")],
        "Passive voice"
    ),

    # ── 11. Coordinating conjunction ─────────────────────────────────────────
    (
        "John and Mary went to the store",
        [("John","and"),("John","Mary"),("went","John"),
         ("store","the"),("to","store"),("went","to"),("ROOT","went")],
        "Coordination: conj NP"
    ),

    # ── 12. Prepositional chain ───────────────────────────────────────────────
    (
        "She walked into the room with a smile",
        [("walked","She"),("room","the"),("into","room"),
         ("walked","into"),("smile","a"),("with","smile"),
         ("walked","with"),("ROOT","walked")],
        "PP chain: two prepositional phrases"
    ),

    # ── 13. Auxiliary verbs ───────────────────────────────────────────────────
    (
        "They should have been told the truth",
        None,
        "Auxiliaries: modal + perfect + passive"
    ),

    # ── 14. Possessive ────────────────────────────────────────────────────────
    (
        "The dog bit the man",
        [("dog","The"),("bit","dog"),("man","the"),("bit","man"),("ROOT","bit")],
        "Basic: definite NPs subject and object"
    ),

    # ── 15. Adverbial modification ────────────────────────────────────────────
    (
        "She spoke very quietly in the corner",
        [("spoke","She"),("quietly","very"),("spoke","quietly"),
         ("corner","the"),("in","corner"),("spoke","in"),("ROOT","spoke")],
        "Adverb modifying adverb + PP"
    ),

    # ── 16. This is a message sentence ───────────────────────────────────────
    (
        "This is a message to all the beings in this galaxy",
        [("is","This"),("message","a"),("beings","all"),("beings","the"),
         ("galaxy","this"),("in","galaxy"),("beings","in"),("to","beings"),
         ("message","to"),("is","message"),("ROOT","is")],
        "Existential + nested PP"
    ),

    # ── 17. Question word order ───────────────────────────────────────────────
    (
        "What did you say to her",
        None,
        "Question: inverted auxiliary"
    ),

    # ── 18. Triple duplicate word ─────────────────────────────────────────────
    (
        "All the waste in a year from a nuclear power plant can be stored under a desk",
        None,
        "Stress: 'a' appears 3 times (duplicate stress test, no gold)"
    ),

    # ── 19. Single word (edge case) ───────────────────────────────────────────
    (
        "Run",
        [("ROOT","Run")],
        "Edge: single word"
    ),

    # ── 20. Two words ─────────────────────────────────────────────────────────
    (
        "Dogs bark",
        [("bark","Dogs"),("ROOT","bark")],
        "Edge: two words"
    ),
]

# ── Colours ───────────────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[1m"; D = "\033[2m"; X = "\033[0m"

# ── Runner ────────────────────────────────────────────────────────────────────
def run(sentence):
    words  = sentence.split()
    state  = PartialParse(words)
    model  = NeuralParserOracle()
    model.build_gold_arcs(sentence)
    step   = 0
    while not state.is_complete and step <= 2 * len(words) + 5:
        t = model.predict_transition(state)
        if t == "SHIFT" and not state.buffer: t = "RIGHT-ARC"
        if   t == "SHIFT":     state.shift()
        elif t == "LEFT-ARC":  state.left_arc()
        elif t == "RIGHT-ARC": state.right_arc()
        step += 1
    return state, step

def dep_tree_lines(state):
    children = defaultdict(list)
    for h, d in state.dependencies:
        children[h].append(d)
    for kids in children.values(): kids.sort()
    lines = ["ROOT"]
    def render(idx, prefix, is_last):
        lines.append(prefix + ("└── " if is_last else "├── ") + state.word(idx))
        kids = children.get(idx, [])
        for i, c in enumerate(kids):
            render(c, prefix + ("    " if is_last else "│   "), i == len(kids)-1)
    for i, c in enumerate(children.get(ROOT_IDX, [])):
        render(c, "", i == len(children[ROOT_IDX])-1)
    return lines

def check(sentence, gold, desc):
    state, steps = run(sentence)
    words = sentence.split()
    deps  = [(state.word(h), state.word(d)) for h, d in state.dependencies]

    stalled    = not state.is_complete
    dups       = len(deps) != len(set(deps))
    covered    = len(deps) == len(words)
    wrong      = [d for d in deps if d not in gold] if gold else None
    missing    = [d for d in gold if d not in deps] if gold else None
    arc_ok     = gold is None or (not wrong and not missing)
    ok         = not stalled and not dups and covered and arc_ok

    return dict(ok=ok, sentence=sentence, desc=desc, words=words, steps=steps,
                deps=deps, stalled=stalled, dups=dups, covered=covered,
                wrong=wrong, missing=missing, gold=gold, state=state)

def print_result(r, idx):
    status = f"{G}✓ PASS{X}" if r["ok"] else f"{R}✗ FAIL{X}"
    print(f"\n{B}[{idx:02d}] {status}  {r['desc']}{X}")
    print(f'     {D}"{r["sentence"]}"{X}')
    print(f"     {D}{len(r['words'])} words · {r['steps']} steps · {len(r['deps'])} deps{X}")

    if r["stalled"]:   print(f"     {R}STALLED — did not complete{X}")
    if r["dups"]:
        seen, dl = set(), []
        for d in r["deps"]:
            if d in seen: dl.append(d)
            seen.add(d)
        print(f"     {R}DUPLICATES: {dl}{X}")
    if not r["covered"]:
        print(f"     {R}Coverage: {len(r['deps'])}/{len(r['words'])} words have a head{X}")
    if r["wrong"]:     print(f"     {R}Wrong arcs:   {r['wrong']}{X}")
    if r["missing"]:   print(f"     {R}Missing arcs: {r['missing']}{X}")
    if r["gold"] and not r["wrong"] and not r["missing"]:
        print(f"     {G}All arcs match gold solution{X}")
    if r["gold"] is None:
        print(f"     {Y}Structural checks only (no gold){X}")

    print(f"     {D}", end="")
    for line in dep_tree_lines(r["state"]):
        print(f"\n       {line}", end="")
    print(X)

def main():
    print(f"\n{B}{'='*65}")
    print("  Dependency Parser — Verification Suite")
    print(f"{'='*65}{X}")

    results = [check(s, g, d) for s, g, d in TESTS]
    for i, r in enumerate(results, 1):
        print_result(r, i)

    passed = sum(1 for r in results if r["ok"])
    total  = len(results)
    colour = G if passed == total else (Y if passed >= total * 0.8 else R)
    print(f"\n{B}{'='*65}")
    print(f"  {colour}{passed}/{total} passed{X}{B}")
    print(f"{'='*65}{X}\n")

if __name__ == "__main__":
    main()
