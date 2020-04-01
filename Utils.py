#!/usr/bin/env python
import numpy as np

from typing import Dict, Generator, List, Optional, Tuple

#From BioPython Package
def FastqGeneralIterator(source):
    """Iterate over Fastq records as string tuples (not as SeqRecord objects).
    Arguments:
     - source - input stream opened in text mode, or a path to a file
    This code does not try to interpret the quality string numerically.  It
    just returns tuples of the title, sequence and quality as strings.  For
    the sequence and quality, any whitespace (such as new lines) is removed.
    Our SeqRecord based FASTQ iterators call this function internally, and then
    turn the strings into a SeqRecord objects, mapping the quality string into
    a list of numerical scores.  If you want to do a custom quality mapping,
    then you might consider calling this function directly.
    For parsing FASTQ files, the title string from the "@" line at the start
    of each record can optionally be omitted on the "+" lines.  If it is
    repeated, it must be identical.
    The sequence string and the quality string can optionally be split over
    multiple lines, although several sources discourage this.  In comparison,
    for the FASTA file format line breaks between 60 and 80 characters are
    the norm.
    **WARNING** - Because the "@" character can appear in the quality string,
    this can cause problems as this is also the marker for the start of
    a new sequence.  In fact, the "+" sign can also appear as well.  Some
    sources recommended having no line breaks in the  quality to avoid this,
    but even that is not enough, consider this example::
        @071113_EAS56_0053:1:1:998:236
        TTTCTTGCCCCCATAGACTGAGACCTTCCCTAAATA
        +071113_EAS56_0053:1:1:998:236
        IIIIIIIIIIIIIIIIIIIIIIIIIIIIICII+III
        @071113_EAS56_0053:1:1:182:712
        ACCCAGCTAATTTTTGTATTTTTGTTAGAGACAGTG
        +
        @IIIIIIIIIIIIIIICDIIIII<%<6&-*).(*%+
        @071113_EAS56_0053:1:1:153:10
        TGTTCTGAAGGAAGGTGTGCGTGCGTGTGTGTGTGT
        +
        IIIIIIIIIIIICIIGIIIII>IAIIIE65I=II:6
        @071113_EAS56_0053:1:3:990:501
        TGGGAGGTTTTATGTGGA
        AAGCAGCAATGTACAAGA
        +
        IIIIIII.IIIIII1@44
        @-7.%<&+/$/%4(++(%
    This is four PHRED encoded FASTQ entries originally from an NCBI source
    (given the read length of 36, these are probably Solexa Illumina reads where
    the quality has been mapped onto the PHRED values).
    This example has been edited to illustrate some of the nasty things allowed
    in the FASTQ format.  Firstly, on the "+" lines most but not all of the
    (redundant) identifiers are omitted.  In real files it is likely that all or
    none of these extra identifiers will be present.
    Secondly, while the first three sequences have been shown without line
    breaks, the last has been split over multiple lines.  In real files any line
    breaks are likely to be consistent.
    Thirdly, some of the quality string lines start with an "@" character.  For
    the second record this is unavoidable.  However for the fourth sequence this
    only happens because its quality string is split over two lines.  A naive
    parser could wrongly treat any line starting with an "@" as the beginning of
    a new sequence!  This code copes with this possible ambiguity by keeping
    track of the length of the sequence which gives the expected length of the
    quality string.
    Using this tricky example file as input, this short bit of code demonstrates
    what this parsing function would return:
    >>> with open("Quality/tricky.fastq") as handle:
    ...     for (title, sequence, quality) in FastqGeneralIterator(handle):
    ...         print(title)
    ...         print("%s %s" % (sequence, quality))
    ...
    071113_EAS56_0053:1:1:998:236
    TTTCTTGCCCCCATAGACTGAGACCTTCCCTAAATA IIIIIIIIIIIIIIIIIIIIIIIIIIIIICII+III
    071113_EAS56_0053:1:1:182:712
    ACCCAGCTAATTTTTGTATTTTTGTTAGAGACAGTG @IIIIIIIIIIIIIIICDIIIII<%<6&-*).(*%+
    071113_EAS56_0053:1:1:153:10
    TGTTCTGAAGGAAGGTGTGCGTGCGTGTGTGTGTGT IIIIIIIIIIIICIIGIIIII>IAIIIE65I=II:6
    071113_EAS56_0053:1:3:990:501
    TGGGAGGTTTTATGTGGAAAGCAGCAATGTACAAGA IIIIIII.IIIIII1@44@-7.%<&+/$/%4(++(%
    Finally we note that some sources state that the quality string should
    start with "!" (which using the PHRED mapping means the first letter always
    has a quality score of zero).  This rather restrictive rule is not widely
    observed, so is therefore ignored here.  One plus point about this "!" rule
    is that (provided there are no line breaks in the quality sequence) it
    would prevent the above problem with the "@" character.
    """
    try:
        handle = open(source)
    except TypeError:
        handle = source
        if handle.read(0) != "":
            raise StreamModeError("Fastq files must be opened in text mode") from None
    try:
        try:
            line = next(handle)
        except StopIteration:
            return  # Premature end of file, or just empty?

        while True:
            if line[0] != "@":
                raise ValueError(
                    "Records in Fastq files should start with '@' character"
                )
            title_line = line[1:].rstrip()
            seq_string = ""
            # There will now be one or more sequence lines; keep going until we
            # find the "+" marking the quality line:
            for line in handle:
                if line[0] == "+":
                    break
                seq_string += line.rstrip()
            else:
                if seq_string:
                    raise ValueError("End of file without quality information.")
                else:
                    raise ValueError("Unexpected end of file")
            # The title here is optional, but if present must match!
            second_title = line[1:].rstrip()
            if second_title and second_title != title_line:
                raise ValueError("Sequence and quality captions differ.")
            # This is going to slow things down a little, but assuming
            # this isn't allowed we should try and catch it here:
            if " " in seq_string or "\t" in seq_string:
                raise ValueError("Whitespace is not allowed in the sequence.")
            seq_len = len(seq_string)

            # There will now be at least one line of quality data, followed by
            # another sequence, or EOF
            line = None
            quality_string = ""
            for line in handle:
                if line[0] == "@":
                    # This COULD be the start of a new sequence. However, it MAY just
                    # be a line of quality data which starts with a "@" character.  We
                    # should be able to check this by looking at the sequence length
                    # and the amount of quality data found so far.
                    if len(quality_string) >= seq_len:
                        # We expect it to be equal if this is the start of a new record.
                        # If the quality data is longer, we'll raise an error below.
                        break
                    # Continue - its just some (more) quality data.
                quality_string += line.rstrip()
            else:
                if line is None:
                    raise ValueError("Unexpected end of file")
                line = None

            if seq_len != len(quality_string):
                raise ValueError(
                    "Lengths of sequence and quality values differs for %s (%i and %i)."
                    % (title_line, seq_len, len(quality_string))
                )

            # Return the record and then continue...
            yield (title_line, seq_string, quality_string)

            if line is None:
                break
    finally:
        if handle is not source:
            handle.close()

def get_fastq_file_handle(fq_file):
    if fq_file.endswith('gz'):
        import gzip
        handle = gzip.open(fq_file, mode='rt')
    elif fq_file.endswith('bz2'):
        import bz2
        handle = bz2.open(fq_file, mode='rt')
    else:
        handle = open(fq_file, 'r')
    return handle

class StreamModeError(ValueError):
    """Incorrect stream mode (text vs binary).
    This error should be raised when a stream (file or file-like object)
    argument is in text mode while the receiving function expects binary mode,
    or vice versa.
    """

class BiopythonWarning(Warning):
    """Biopython warning.
    Biopython should use this warning (or subclasses of it), making it easy to
    silence all our warning messages should you wish to:
    >>> import warnings
    >>> from Bio import BiopythonWarning
    >>> warnings.simplefilter('ignore', BiopythonWarning)
    Consult the warnings module documentation for more details.
    """