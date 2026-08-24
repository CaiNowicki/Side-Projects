# Spanish Word Frequency List

`Spanish_1000_Word_Frequency_List.xlsx` — the 1000 most common Spanish words with English
translations, grouped into topical sections and sorted alphabetically (by the Spanish word)
within each section. The "Word List" sheet lays the full list out across three side-by-side
Spanish/English column pairs; the "Overview" sheet explains the methodology and lists topic
sizes.

**Methodology:** compiled from an LLM's trained knowledge of Spanish vocabulary, shaped by
published Spanish frequency research (RAE corpora, Davies' *A Frequency Dictionary of
Spanish*) — not a live statistical frequency count run against a specific corpus, since no
such corpus-analysis tool was executed for this task.

## Regenerating

```
python3 -m pip install openpyxl
python3 build.py
```

`data.py` holds the word list (grouped by topic); `build.py` dedupes, caps each topic to its
target size, alphabetizes, and writes the formatted workbook.
