# -*- coding: utf-8 -*-
import unicodedata
from datetime import date
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from data import TOPICS

TARGET_TOTAL = 1000
FONT_NAME = "Arial"

# ---------------------------------------------------------------------------
# 1. Deduplicate (case-insensitive, keep first occurrence) and trim to exactly
#    TARGET_TOTAL words by removing entries from the tail of the currently
#    largest topic, one at a time, until the total matches.
# ---------------------------------------------------------------------------
topics = {name: list(words) for name, words in TOPICS.items()}

seen = set()
for name, words in topics.items():
    deduped = []
    for es, en in words:
        key = es.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((es, en))
    topics[name] = deduped

total = sum(len(w) for w in topics.values())
if total < TARGET_TOTAL:
    raise SystemExit(f"Only {total} unique words available, need {TARGET_TOTAL}")

# Explicit per-topic caps, chosen so the mix reflects real Spanish word
# frequency (many core verbs/adjectives/adverbs survive; small closed-class
# categories like pronouns/prepositions are left almost untouched) rather
# than mechanically shrinking whichever category happens to be largest.
TARGET_SIZE = {
    "Pronouns": 34,
    "Articles & Determiners": 15,
    "Conjunctions": 20,
    "Prepositions": 25,
    "Question Words": 12,
    "Numbers": 35,
    "Time & Calendar": 45,
    "Adverbs": 52,
    "Adjectives - Descriptive": 95,
    "Verbs - Core Actions": 130,
    "Family & People": 34,
    "Body & Health": 42,
    "Emotions & Mind": 32,
    "Food & Drink": 55,
    "Colors": 13,
    "Clothing": 20,
    "Home & Household": 42,
    "Nature & Weather": 40,
    "Animals": 30,
    "Places, Geography & Directions": 36,
    "Transportation & Travel": 28,
    "Society, Government & Law": 28,
    "Work, Business & Money": 30,
    "Education & Language": 22,
    "Technology & Communication": 22,
    "Sports, Leisure & Entertainment": 20,
    "Abstract Concepts & Miscellaneous Nouns": 43,
}
assert sum(TARGET_SIZE.values()) == TARGET_TOTAL, sum(TARGET_SIZE.values())
assert set(TARGET_SIZE) == set(topics), set(TARGET_SIZE) ^ set(topics)

for name, cap in TARGET_SIZE.items():
    available = len(topics[name])
    if available < cap:
        raise SystemExit(f"{name}: only {available} unique words, need {cap}")
    topics[name] = topics[name][:cap]

assert sum(len(w) for w in topics.values()) == TARGET_TOTAL

# ---------------------------------------------------------------------------
# 2. Sort alphabetically by Spanish word within each topic (accent-insensitive).
# ---------------------------------------------------------------------------
def sort_key(word):
    nfkd = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))

for name in topics:
    topics[name].sort(key=lambda pair: sort_key(pair[0]))

# ---------------------------------------------------------------------------
# 3. Build the master ordered row sequence: topic header rows interleaved
#    with (spanish, english) word rows, in topic order.
# ---------------------------------------------------------------------------
TOPIC_ORDER = list(TOPICS.keys())

rows = []  # list of ("header", topic_name) or ("word", es, en)
for name in TOPIC_ORDER:
    rows.append(("header", name, None))
    for es, en in topics[name]:
        rows.append(("word", es, en))

total_rows = len(rows)
print(f"Total word rows: {sum(1 for r in rows if r[0] == 'word')}")
print(f"Total rows incl. headers: {total_rows}")

# ---------------------------------------------------------------------------
# 4. Fold into 3 side-by-side column pairs (newspaper-style flow).
# ---------------------------------------------------------------------------
import math
rows_per_col = math.ceil(total_rows / 3)
chunks = [rows[i:i + rows_per_col] for i in range(0, total_rows, rows_per_col)]
while len(chunks) < 3:
    chunks.append([])

# ---------------------------------------------------------------------------
# 5. Write workbook.
# ---------------------------------------------------------------------------
wb = Workbook()

# --- Overview sheet -------------------------------------------------------
ov = wb.active
ov.title = "Overview"
ov.sheet_view.showGridLines = False

title_font = Font(name=FONT_NAME, size=16, bold=True, color="1F4E78")
h2_font = Font(name=FONT_NAME, size=12, bold=True, color="1F4E78")
body_font = Font(name=FONT_NAME, size=11)
italic_font = Font(name=FONT_NAME, size=10, italic=True, color="595959")

ov.column_dimensions["A"].width = 100
r = 1
ov.cell(row=r, column=1, value="1000 Most Common Spanish Words, by Topic").font = title_font
r += 2
ov.cell(row=r, column=1, value="Methodology").font = h2_font
r += 1
methodology = (
    "This list was compiled using an LLM's (Claude's) trained knowledge of Spanish "
    "vocabulary, informed by the general shape of published Spanish word-frequency "
    "research (e.g. RAE corpora, Mark Davies' 'A Frequency Dictionary of Spanish'). "
    "It is a knowledge-based approximation, not the output of a live statistical "
    "frequency count run against a specific text corpus -- no such corpus-analysis "
    "tool was executed for this task. The ~1000 words selected represent core, "
    "high-frequency Spanish vocabulary spanning function words (pronouns, "
    "prepositions, conjunctions), the most common verbs and adjectives, and "
    "everyday nouns across common life topics."
)
for line in [methodology[i:i+110] for i in range(0, len(methodology), 110)]:
    pass  # not used; wrap via alignment instead

cell = ov.cell(row=r, column=1, value=methodology)
cell.font = body_font
cell.alignment = Alignment(wrap_text=True, vertical="top")
ov.row_dimensions[r].height = 90
r += 2

ov.cell(row=r, column=1, value="Organization").font = h2_font
r += 1
organization = (
    "Words are grouped into topical sections (e.g. Family & People, Food & Drink, "
    "Verbs - Core Actions). Within each topic, words are sorted alphabetically by "
    "the Spanish term (accent-insensitive). The 'Word List' sheet lays the full "
    "topic-by-topic list out across three side-by-side Spanish/English column "
    "pairs, reading top-to-bottom within the first pair, then continuing at the "
    "top of the second pair, then the third -- a compact, newspaper-style layout "
    "for a list this long. A topic's word block may therefore continue from the "
    "bottom of one column pair to the top of the next."
)
cell = ov.cell(row=r, column=1, value=organization)
cell.font = body_font
cell.alignment = Alignment(wrap_text=True, vertical="top")
ov.row_dimensions[r].height = 90
r += 2

ov.cell(row=r, column=1, value="Contents").font = h2_font
r += 1
ov.cell(row=r, column=1, value=f"Total words: {TARGET_TOTAL}").font = body_font
r += 1
ov.cell(row=r, column=1, value=f"Topics: {len(TOPIC_ORDER)}").font = body_font
r += 2
ov.cell(row=r, column=1, value="Topic sections (in list order):").font = h2_font
r += 1
for name in TOPIC_ORDER:
    ov.cell(row=r, column=1, value=f"  • {name}  ({len(topics[name])} words)").font = body_font
    r += 1
r += 1
ov.cell(row=r, column=1, value=f"Generated {date.today().isoformat()}.").font = italic_font

# --- Word List sheet -------------------------------------------------------
ws = wb.create_sheet("Word List")
ws.sheet_view.showGridLines = False

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(name=FONT_NAME, size=11, bold=True, color="FFFFFF")
TOPIC_FILL = PatternFill("solid", fgColor="DDEBF7")
TOPIC_FONT = Font(name=FONT_NAME, size=11, bold=True, color="1F4E78")
WORD_ES_FONT = Font(name=FONT_NAME, size=11, bold=True)
WORD_EN_FONT = Font(name=FONT_NAME, size=11)
thin = Side(style="thin", color="D9D9D9")
CELL_BORDER = Border(bottom=thin)

col_pairs = [(1, 2), (3, 4), (5, 6)]  # (A,B) (C,D) (E,F)

# Header row (row 1) for each pair
for (c_es, c_en) in col_pairs:
    for c, label in ((c_es, "Spanish"), (c_en, "English")):
        cell = ws.cell(row=1, column=c, value=label)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.column_dimensions[get_column_letter(c_es)].width = 22
    ws.column_dimensions[get_column_letter(c_en)].width = 34

ws.freeze_panes = "A2"
ws.row_dimensions[1].height = 20

for chunk, (c_es, c_en) in zip(chunks, col_pairs):
    for i, entry in enumerate(chunk):
        row_num = i + 2  # data starts at row 2, under the header row
        kind = entry[0]
        if kind == "header":
            topic_name = entry[1]
            ws.merge_cells(start_row=row_num, start_column=c_es, end_row=row_num, end_column=c_en)
            cell = ws.cell(row=row_num, column=c_es, value=topic_name)
            cell.font = TOPIC_FONT
            cell.fill = TOPIC_FILL
            cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)
            other = ws.cell(row=row_num, column=c_en)
            other.fill = TOPIC_FILL
        else:
            _, es, en = entry
            c1 = ws.cell(row=row_num, column=c_es, value=es)
            c1.font = WORD_ES_FONT
            c1.border = CELL_BORDER
            c2 = ws.cell(row=row_num, column=c_en, value=en)
            c2.font = WORD_EN_FONT
            c2.border = CELL_BORDER

ws.print_title_rows = "1:1"

wb.save("Spanish_1000_Word_Frequency_List.xlsx")
print("Saved Spanish_1000_Word_Frequency_List.xlsx")
