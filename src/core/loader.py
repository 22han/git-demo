"""
PDF Loader Module
==================
Loads PDF files, extracts text, splits into chunks using TokenTextSplitter,
and prepends filename metadata to each chunk for precise retrieval.

Filename pattern: {COMPANY}_{YEAR}_{TYPE}.pdf
Example: 3M_2018_10K.pdf -> company="3M", year="2018", type="10K"
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class PDFLoader:
    """Loader for PDF files with metadata extraction and text chunking."""

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the PDF loader.

        Args:
            data_dir: Directory containing PDF files.
            chunk_size: Number of tokens per chunk.
            chunk_overlap: Number of overlapping tokens between chunks.
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse PDF filename to extract metadata.

        Expected pattern: {COMPANY}_{YEAR}_{TYPE}.pdf
        Examples:
            3M_2018_10K.pdf         -> company="3M", year="2018", type="10K"
            AMAZON_2019_10K.pdf     -> company="AMAZON", year="2019", type="10K"
            AMCOR_2022_8K.pdf       -> company="AMCOR", year="2022", type="8K"
            AMCOR_2023_Q4_EARNINGS.pdf -> company="AMCOR", year="2023", type="Q4_EARNINGS"
            AMCOR_2022_8K_dated-2022-07-01.pdf -> company="AMCOR", year="2022", type="8K"

        Args:
            filename: Name of the PDF file.

        Returns:
            Dictionary with keys: company, year, type, original_filename.
        """
        # Remove .pdf extension
        name_without_ext = os.path.splitext(filename)[0]

        # Split by underscore
        parts = name_without_ext.split("_")

        # Default values
        company = ""
        year = ""
        doc_type = ""

        if len(parts) >= 2:
            # Try to find year (4-digit number) in the parts
            year_pattern = re.compile(r"^\d{4}$")
            year_index = None

            for i, part in enumerate(parts):
                if year_pattern.match(part):
                    year = part
                    year_index = i
                    break

            if year_index is not None:
                # Company is everything before the year part
                company = "_".join(parts[:year_index])
                # Type is everything after the year part
                type_parts = parts[year_index + 1:]
                # Handle special cases like "dated-2022-07-01"
                cleaned_type_parts = []
                for tp in type_parts:
                    # Skip "dated" and date-like patterns (YYYY-MM-DD)
                    if tp == "dated" or re.match(r"^\d{2}-\d{2}-\d{2}$", tp):
                        continue
                    cleaned_type_parts.append(tp)
                doc_type = "_".join(cleaned_type_parts) if cleaned_type_parts else "unknown"
            else:
                # No year found, assume first part is company, rest is type
                company = parts[0]
                doc_type = "_".join(parts[1:])

        return {
            "company": company,
            "year": year,
            "type": doc_type,
            "original_filename": filename,
        }

    def build_metadata_prefix(self, metadata: Dict[str, str]) -> str:
        """
        Build a metadata prefix string to prepend to each chunk.

        Args:
            metadata: Dictionary with keys: company, year, type, original_filename.

        Returns:
            Formatted metadata prefix string.
        """
        prefix_parts = []
        if metadata["company"]:
            prefix_parts.append(f"Company: {metadata['company']}")
        if metadata["year"]:
            prefix_parts.append(f"Year: {metadata['year']}")
        if metadata["type"]:
            prefix_parts.append(f"Type: {metadata['type']}")
        prefix_parts.append(f"File: {metadata['original_filename']}")

        return f"[{' | '.join(prefix_parts)}]\n\n"

    def load_single_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Load a single PDF file, split into chunks, and prepend metadata.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of dictionaries, each containing:
                - text: chunk text with metadata prefix
                - metadata: original metadata dictionary
                - page: page number (if available)
        """
        filename = os.path.basename(pdf_path)
        metadata = self.parse_filename(filename)
        prefix = self.build_metadata_prefix(metadata)

        # Load PDF using PyPDFLoader
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
            return []

        chunks = []
        current_chunk_text = ""
        current_chunk_pages = []
        token_count = 0

        # Process each page - split by page first, then merge across pages
        page_texts = []
        for page in pages:
            page_text = page.page_content
            page_num = page.metadata.get("page", 0)
            page_texts.append((page_text, page_num))

        # Combine all text and split
        full_text = "\n".join([pt[0] for pt in page_texts])

        # Split into chunks using TokenTextSplitter
        split_chunks = self.splitter.split_text(full_text)

        # For each chunk, try to find which page it belongs to
        # Build character-to-page mapping
        char_page_map = []
        char_pos = 0
        for page_text, page_num in page_texts:
            char_page_map.append((char_pos, char_pos + len(page_text), page_num))
            char_pos += len(page_text) + 1  # +1 for the newline we added

        for chunk_text in split_chunks:
            # Find the starting position of this chunk in the full text
            start_pos = full_text.find(chunk_text)
            if start_pos == -1:
                # Try to find a substring match
                # Fallback: use the first page
                chunk_page = 0
            else:
                # Find which page this chunk starts on
                chunk_page = 0
                for char_start, char_end, page_num in char_page_map:
                    if char_start <= start_pos < char_end:
                        chunk_page = page_num
                        break

            # Create the chunk with metadata prefix
            chunk_with_metadata = {
                "text": prefix + chunk_text,
                "metadata": metadata,
                "page": chunk_page,
            }
            chunks.append(chunk_with_metadata)

        return chunks

    def load_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Load all PDF files from the data directory.

        Returns:
            List of all chunks from all PDF files.
        """
        all_chunks = []
        pdf_files = self.get_pdf_files()

        print(f"Found {len(pdf_files)} PDF files in {self.data_dir}")

        for i, pdf_path in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_path)
            print(f"[{i}/{len(pdf_files)}] Processing: {filename}")

            chunks = self.load_single_pdf(pdf_path)
            all_chunks.extend(chunks)

            print(f"  -> Generated {len(chunks)} chunks")

        print(f"\nTotal chunks generated: {len(all_chunks)}")
        return all_chunks

    def get_pdf_files(self) -> List[str]:
        """
        Get all PDF files in the data directory (non-recursive).

        Returns:
            Sorted list of PDF file paths.
        """
        pdf_files = []
        data_path = Path(self.data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        for file in sorted(data_path.iterdir()):
            if file.is_file() and file.suffix.lower() == ".pdf":
                pdf_files.append(str(file))

        return pdf_files


def main():
    """Main function to demonstrate the PDF loader."""
    import json

    # Configure paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = r"D:\finance_rag\finance_rag_evaluation\data"


    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please create a 'data' directory and place your PDF files there.")
        return

    # Initialize loader
    loader = PDFLoader(
        data_dir=data_dir,
        chunk_size=512,
        chunk_overlap=50,
    )

    # Load all PDFs
    chunks = loader.load_all_pdfs()

    # Print sample chunks
    if chunks:
        print("\n" + "=" * 80)
        print("SAMPLE CHUNKS (first 3)")
        print("=" * 80)

        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Metadata: {chunk['metadata']}")
            print(f"Page: {chunk['page']}")
            print(f"Text preview (first 300 chars):")
            print(chunk['text'][:50])
            print("...")
    else:
        print("No chunks were generated.")


if __name__ == "__main__":
    main()