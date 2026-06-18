"""
context — the law knowledge and situational awareness behind the commentary.

Ties the project together: events enter from data_replayer, the Granite-powered
commentary agent turns them into text (lead + analyst), and a speaker optionally
turns that text into audio. Going live later is a source swap, not a rewrite.

Modules
-------
ingest_laws.py
    ingest_laws.py parses FIFA laws of the game and creates chunks for ingestion into a vector database.

The code uses the IBM Docling DocumentConverter to convert the PDF document into a format suitable for chunking. 
Then, it uses the HybridChunker to create contextualized chunks from the document. 
These chunks can then be ingested into a vector database for further processing and analysis.
"""