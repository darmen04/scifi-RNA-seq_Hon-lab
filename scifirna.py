#!/usr/bin/env python

import argparse
import datetime
import Tools

parser = argparse.ArgumentParser()

parser.add_argument('mode', help="""Mode: one of "all", "preproc", "star", or "postproc".
"all" runs the entire pipeline.
"preproc" or "preprocess" runs all the steps prior to running STAR, producing a file called single_cells_barcoded_head.fastq in output_dir.
"star" assumes that output of preproc exists in output_dir, and tries to run the STAR alignment, producing a file called single_cells_barcoded_headAligned.out.sam.
"postproc" assumes that the output of star exists in  output_dir.
""")

parser.add_argument('--fq1', help='fastq1 - reads contain UMI and RT barcodes')
parser.add_argument('--fq2', help='fastq2 - reads contain 10x barcodes')
parser.add_argument('--fq3', help='fastq3 - mRNA reads')
parser.add_argument('--output_dir', help='output dir')
parser.add_argument('--chemistry', default='scifi_v1', help='Using v1 or v2 chemistry')
parser.add_argument('--genome_dir', default='./', help='path containing reference genome')
parser.add_argument('--nthreads', default = '2', help='Number of threads')

args = parser.parse_args()

mode = args.mode.lower()
if mode == 'mkref':
    # Generate genome
    Tools.make_combined_genome(args.genome, args.fasta, args.output_dir)
    
    # Make a combine annotation from GTF
    Tools.make_gtf_annotations(args.genome, args.genes, args.output_dir, args.splicing)
    
    # Index genome with star
    Tools.generate_STAR_index(args.output_dir, args.nthreads, args.genomeSAindexNbases, args.splicing)

if mode == 'all':
    #Preprocess the fastq files
    Tools.preprocess_fastq(args.fq1, args.fq2, args.fq3, args.output_dir, args.nthreads, args.chemistry)

    #Mapping by using STAR
    Tools.run_star(args.genome_dir, args.output_dir, args.nthreads)
    Tools.sort_sam(args.output_dir, args.nthreads)

    #Count the reads

    #Call cells

