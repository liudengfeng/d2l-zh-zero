import bibtexparser
import os


def read_bib_file(bib_file_path):
    with open(bib_file_path, "r", encoding="utf-8") as bib_file:
        bib_database = bibtexparser.load(bib_file)
    return bib_database.entries


def format_bib_entry(entry):
    authors = (
        entry.get("author", "Unknown Author").replace("\n", " ").replace(" and ", ", ")
    )
    title = entry.get("title", "No Title")
    journal = entry.get("journal", "")
    year = entry.get("year", "No Year")
    volume = entry.get("volume", "")
    number = entry.get("number", "")
    pages = entry.get("pages", "")
    doi = entry.get("doi", "")

    formatted_entry = f"**{authors}**. *{title}*. {journal}, {year}"
    if volume:
        formatted_entry += f", vol. {volume}"
    if number:
        formatted_entry += f", no. {number}"
    if pages:
        formatted_entry += f", pp. {pages}"
    if doi:
        formatted_entry += f". DOI: [{doi}](https://doi.org/{doi})"

    return formatted_entry


def convert_bib_to_md(bib_file_path, md_file_path):
    entries = read_bib_file(bib_file_path)
    with open(md_file_path, "w", encoding="utf-8") as md_file:
        for entry in entries:
            formatted_entry = format_bib_entry(entry)
            md_file.write(formatted_entry + "\n\n")


if __name__ == "__main__":
    bib_file_path = r"C:\Users\liude\Documents\github\d2l-zh-new\course\d2l.bib"
    md_file_path = r"C:\Users\liude\Documents\github\d2l-zh-new\course\17-参考文献\zreferences.md"
    convert_bib_to_md(bib_file_path, md_file_path)
