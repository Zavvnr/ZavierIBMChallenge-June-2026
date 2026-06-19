# context/laws/ — the IFAB Laws of the Game source

Drop the IFAB Laws of the Game PDF in this folder, named exactly:

    laws_of_the_game.pdf

Then build the local knowledge base from the repo root:

    python -m context.ingest_laws

(You can skip the rename/move and point the ingester straight at the download instead:
`python -m context.ingest_laws --pdf "C:\path\to\the-downloaded-file.pdf"`.)

## Where to get it

Current edition: **Laws of the Game 2025/26** (effective 1 July 2025), free from The IFAB.

- Documents page (all languages/formats): https://www.theifab.com/laws-of-the-game-documents/
- Direct, English, **single-page** format (parses best with Docling):
  https://downloads.theifab.com/downloads/laws-of-the-game-2025-26-single-pages?l=en

Prefer the **single-page** version over the double-page spread — Docling reads the
single-page layout more cleanly (the spread interleaves two columns).

## Notes

- This PDF is third-party content — keep it out of git. Add `context/laws/*.pdf` (and the
  generated `context/index/`) to `.gitignore`.
- Attribution: © The IFAB. Used here for non-commercial, educational/competition purposes.
- Ingesting embeds each chunk with Granite, so a **Granite embedding model must be loaded
  in LM Studio** and `GRANITE_BASE_URL` set before you run it.
