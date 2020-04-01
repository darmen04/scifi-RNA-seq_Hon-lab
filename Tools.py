#! /usr/bin/env python
import os
import subprocess
import sys
import gzip
import pickle
import warnings
import Utils
import itertools

import pandas as pd
import numpy as np

from numpy import unique
from numba import jit
from collections import defaultdict
from typing import List, Dict, Generator, Tuple
from Utils import FastqGeneralIterator, get_fastq_file_handle
from multiprocessing import Pool, Value

PATH = os.path.dirname(__file__)
HOME = os.path.expanduser('~')

STAR_PATH = os.path.join(HOME, 'split_seq_reqs', 'bin', 'STAR')
if not os.path.exists(STAR_PATH):
    STAR_PATH = 'STAR'

SAMTOOLS_PATH = os.path.join(HOME, 'split_seq_reqs', 'bin', 'samtools')
if not os.path.exists(SAMTOOLS_PATH):
    SAMTOOLS_PATH = 'samtools'

@jit
def twobit_to_dna(twobit: int, size: int) -> str:
	result = []
	for i in range(size):
		x = (twobit & (3 << 2 * i)) >> 2 * i
		if x == 0:
			result.append("A")
		elif x == 1:
			result.append("C")
		elif x == 2:
			result.append("G")
		elif x == 3:
			result.append("T")
	result.reverse()
	return "".join(result)

@jit
def dna_to_twobit(dna: str) -> int:
    x = 0
    for nt in dna:
        if nt == "A":
            x += 0
        elif nt == "C":
            x += 1
        elif nt == "G":
            x += 2
        elif nt == "T":
            x += 3
        x <<= 2
    x >>= 2
    return x

@jit
def twobit_1hamming(twobit: int, size: int) -> List[int]:
	result = []
	for i in range(size):
		x = (twobit >> 2 * (size - i - 1)) & 3
		for j in range(4):
			if x == j:
				continue
			result.append(twobit & ~(3 << 2 * (size - i - 1)) | (j << 2 * (size - i - 1)))
	return result

def correct(query_string: str, whitelist_twobit, bc_len: int):
    if whitelist_twobit != []:
        corrected = False
        for mut in twobit_1hamming(dna_to_twobit(query_string), size=bc_len):
            if mut in whitelist_twobit:
                query_string = twobit_to_dna(mut, bc_len)
                corrected = True
                break
        return query_string, corrected

def barcode_correction_helper(items, pos_tuple, whitelist_bc1, whitelist_bc2, whitelist_bc1_twobit, whitelist_bc2_twobit):
    r1, r2, r3_header, r3, r3_qual = items
    umi_starts, bc_starts = pos_tuple

    umi_seq = r1[umi_starts:(umi_starts+8)]
    bc1_seq = r1[bc_starts:(bc_starts+11)]

    bc1_len = len(whitelist_bc1[0])
    bc2_len = len(whitelist_bc2[0])

    #return the barcodes as is if perfect match
    if (bc1_seq in whitelist_bc1) & (r2[0:bc2_len] in whitelist_bc2):
        corrected_header = ("_").join(["@", bc1_seq, r2[0:bc2_len], umi_seq, r3_header])
        return (corrected_header, r3, r3_qual)
    #otherwise try to correct the barcodes
    else:

        bc1, corrected_1 = correct(bc1_seq, whitelist_bc1_twobit, bc1_len)
        bc2, corrected_2 = correct(r2[0:bc2_len], whitelist_bc2_twobit, bc2_len)
    
        if corrected_1 != False and corrected_2 != False:
            corrected_header = ("_").join(["@", bc1, bc2, umi_seq, r3_header])
            return (corrected_header, r3, r3_qual)
        else:
            return ("","","")

def get_read_sequence(fastq_iter1, fastq_iter2, fastq_iter3):
    for (_x1,y1,_z1), (_x2,y2,_z2), (x3, y3, z3) in zip(fastq_iter1, fastq_iter2, fastq_iter3):
        yield y1, y2, x3, y3, z3

def preprocess_fastq(fastq1, fastq2, fastq3, output_dir, nthreads = int, chemistry='scifi_v1', **params):
    """
    Performs all the steps before running the alignment. Temporary files
    saved in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if chemistry=='scifi_v1':
        bc_whitelist_R1 = pd.read_csv(PATH + '/scifi_bc_384.txt', names=['barcode']).barcode.values
        #bc_whitelist_R2 = pd.read_csv(PATH + '/737K-cratac-v1.txt',names=['barcode']).barcode.values
        bc_whitelist_R2 = pd.read_csv(PATH + '/mock_atac_bc.txt',names=['barcode']).barcode.values

        bc_whitelist_R1_twobit = set()
        bc_whitelist_R2_twobit = set()
        for bc1, bc2 in zip(bc_whitelist_R1, bc_whitelist_R2):
            bc_whitelist_R1_twobit.add(dna_to_twobit(bc1))
            bc_whitelist_R2_twobit.add(dna_to_twobit(bc2))

        # Amplicon sequence
        amp_seq = 'UUUUUUUUNIIIIIIIIIIIV'

        umi_starts = 0
        bc_starts = 0
        for i, c in enumerate(amp_seq):
            if c == 'U':
                umi_starts = i
                break
        for i, c in enumerate(amp_seq):
            if c == 'I':
                bc_starts = i
                break

    #load the fastq files to iterator
    fastq_iter1 = FastqGeneralIterator(get_fastq_file_handle(fastq1))
    fastq_iter2 = FastqGeneralIterator(get_fastq_file_handle(fastq2))
    fastq_iter3 = FastqGeneralIterator(get_fastq_file_handle(fastq3))

    raw_output = os.path.join(output_dir, 'single_cells_barcoded_head.fastq.gz')
    raw_output_fh = gzip.open(raw_output, 'wb')

    counter = Value('i', 0)

    chunk_size = 10000
    items = list(itertools.islice(get_read_sequence(fastq_iter1, fastq_iter2, fastq_iter3), chunk_size))

    with Pool(processes = int(nthreads)) as p:
        while items:
            
            results = p.starmap(barcode_correction_helper, zip(items, 
                                                            itertools.repeat((umi_starts, bc_starts)),
                                                            itertools.repeat(bc_whitelist_R1),
                                                            itertools.repeat(bc_whitelist_R2),
                                                            itertools.repeat(bc_whitelist_R1_twobit),
                                                            itertools.repeat(bc_whitelist_R2_twobit)))
                                                             
            #write to the raw output file.
            for i in results:
                counter.value += 1
                if ((counter.value % 1000 == 0) and (counter.value % 10000 != 0)):
                    print(counter.value, flush=True, file=sys.stdout)
                elif ((counter.value % 100 == 0) and (counter.value % 10000 != 0)):
                    print(".", end='', flush=True, file=sys.stdout)

                if i[0]:
                    raw_output_fh.write((i[0] + '\n').encode('ascii'))
                    raw_output_fh.write((i[1] + '\n').encode('ascii'))
                    raw_output_fh.write(("+" + '\n').encode('ascii'))
                    raw_output_fh.write((i[2] + '\n').encode('ascii'))

            items = list(itertools.islice(get_read_sequence(fastq_iter1, fastq_iter2, fastq_iter3), chunk_size))

    raw_output_fh.close()

def run_star(genome_dir, output_dir, nthreads):
    """ Align reads using STAR.
    """

    nthreads = int(nthreads)
    rc = subprocess.call(STAR_PATH + \
        """ --genomeDir {0}/\
            --runThreadN {2}\
            --readFilesIn {1}/single_cells_barcoded_head.fastq.gz\
            --outFileNamePrefix {1}/single_cells_barcoded_head""".format(genome_dir, output_dir, nthreads),\
             shell=True)
    
    # Add alignment stats to pipeline_stats
    with open(output_dir + '/single_cells_barcoded_headLog.final.out') as f:
        for i in range(8):
            f.readline()
        unique_mapping = int(f.readline().split('\t')[1][:-1])
        for i in range(14):
            f.readline()
        multimapping = int(f.readline().split('\t')[1][:-1])
    with open(output_dir + '/pipeline_stats.txt', 'a') as f:
        f.write('uniquely_aligned\t%d\n' %unique_mapping)
        f.write('multimapping\t%d\n' %multimapping)

    return rc

def sort_sam(output_dir, nthreads):
    """ Sort samfile by header (cell_barcodes, umi) """
    nthreads = int(nthreads)
    rc = subprocess.call(SAMTOOLS_PATH + \
        """ sort -n\
                -@ {1}\
                -T {0}/single_cells_barcoded_headAligned.sort\
                -o {0}/single_cells_barcoded_headAligned.sorted.bam {0}/single_cells_barcoded_headAligned.out.sam""".format(output_dir, nthreads),\
                     shell=True)
    os.remove("""{0}/single_cells_barcoded_headAligned.out.sam""".format(output_dir))
    return rc

#The following function are from split-seq pipeline
#
#https://github.com/Alex-Rosenberg/split-seq-pipeline
#
def make_combined_genome(species, fasta_filenames, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a combined fasta file with species names added to the start of each chromosome name
    cur_fa = fasta_filenames[0]
    cur_species = species[0]
    if fasta_filenames[0].split('.')[-1]=='gz':
        command = """gunzip -cd {0} | awk 'substr($0,1,1)==">"{{print ">{1}_"substr($1,2,length($1)-1),$2,$3,$4}}substr($0,1,1)!=">"{{print $0}}' > {2}/genome.fa""".format(cur_fa, cur_species, output_dir)
    else:
        command = """cat {0} | awk 'substr($0,1,1)==">"{{print ">{1}_"substr($1,2,length($1)-1),$2,$3,$4}}substr($0,1,1)!=">"{{print $0}}' > {2}/genome.fa""".format(cur_fa, cur_species, output_dir)
    rc = subprocess.call(command, shell=True)
    
    for i in range(1,len(species)):
        cur_fa = fasta_filenames[i]
        cur_species = species[i]
        if fasta_filenames[0].split('.')[-1]=='gz':
            command = """gunzip -cd {0} | awk 'substr($0,1,1)==">"{{print ">{1}_"substr($1,2,length($1)-1),$2,$3,$4}}substr($0,1,1)!=">"{{print $0}}' >> {2}/genome.fa""".format(cur_fa, cur_species, output_dir)
        else:
            command = """cat {0} | awk 'substr($0,1,1)==">"{{print ">{1}_"substr($1,2,length($1)-1),$2,$3,$4}}substr($0,1,1)!=">"{{print $0}}' >> {2}/genome.fa""".format(cur_fa, cur_species, output_dir)
        rc = subprocess.call(command, shell=True)
        
def make_gtf_annotations(species, gtf_filenames, output_dir, splicing):
    splicing = splicing=='True'

    # Load the GTFs
    names = ['Chromosome',
         'Source',
         'Feature',
         'Start',
         'End',
         'Score',
         'Strand',
         'Frame',
         'Attributes']

    gtfs = {}
    for i in range(len(species)):
        s = species[i]
        filename = gtf_filenames[i]
        gtfs[s] = pd.read_csv(filename,sep='\t',names=names,comment='#',engine='python')
    
    # TODO: allow users to specify the gene biotypes that they want to keep
    # For now we keep the following
    gene_biotypes_to_keep = ['protein_coding',
                             'lincRNA',
                             'antisense',
                             'IG_C_gene',
                             'IG_C_pseudogene',
                             'IG_D_gene',
                             'IG_J_gene',
                             'IG_J_pseudogene',
                             'IG_V_gene',
                             'IG_V_pseudogene',
                             'TR_C_gene',
                             'TR_D_gene',
                             'TR_J_gene',
                             'TR_J_pseudogene',
                             'TR_V_gene',
                             'TR_V_pseudogene']
    if splicing:
        # Generate a combined GTF with only the gene annotations
        gtf_gene_combined = gtfs[species[0]].query('Feature=="gene"')
        gtf_gene_combined.loc[:,'Chromosome'] = species[0] + '_' + gtf_gene_combined.Chromosome.apply(lambda s:str(s))
        for i in range(1,len(species)):
            gtf_gene_combined_temp = gtfs[species[i]].query('Feature=="gene"')
            gtf_gene_combined_temp.loc[:,'Chromosome'] = species[i] + '_' + gtf_gene_combined_temp.Chromosome.apply(lambda s:str(s))
            gtf_gene_combined = pd.concat([gtf_gene_combined,gtf_gene_combined_temp])
        gene_biotypes = gtf_gene_combined.Attributes.apply(lambda s: get_attribute(s,'gene_biotype'))
        #gtf_gene_combined = gtf_gene_combined.iloc[np.where(gene_biotypes.isin(gene_biotypes_to_keep).values)]
        gtf_gene_combined.index = range(len(gtf_gene_combined))
        gtf_gene_combined.to_csv(output_dir + '/genes.gtf',sep='\t',index=False)
    
    # Generate a combined GTF with only the exon annotations
    gtf_exon_combined = gtfs[species[0]].query('Feature=="exon"')
    gtf_exon_combined.loc[:,'Chromosome'] = species[0] + '_' + gtf_exon_combined.Chromosome.apply(lambda s:str(s))
    for i in range(1,len(species)):
        gtf_exon_combined_temp = gtfs[species[i]].query('Feature=="exon"')
        gtf_exon_combined_temp.loc[:,'Chromosome'] = species[i] + '_' + gtf_exon_combined_temp.Chromosome.apply(lambda s:str(s))
        gtf_exon_combined = pd.concat([gtf_exon_combined,gtf_exon_combined_temp])
    gene_biotypes = gtf_exon_combined.Attributes.apply(lambda s: get_attribute(s,'gene_biotype'))
    #gtf_exon_combined = gtf_exon_combined.iloc[np.where(gene_biotypes.isin(gene_biotypes_to_keep).values)]
    gtf_exon_combined.index = range(len(gtf_exon_combined))
    gtf_exon_combined.to_csv(output_dir + '/exons.gtf',sep='\t',index=False)
    
    if not splicing:
        gtf_gene_combined = gtf_exon_combined.copy(deep=True)
        gtf_gene_combined['Feature'] = 'gene'
        gtf_gene_combined.to_csv(output_dir + '/genes.gtf',sep='\t',index=False)
    # Get locations of genes. We are using the longest possible span of different transcripts here
    gtf_gene_combined.loc[:,'gene_id'] = gtf_gene_combined.Attributes.apply(lambda s: get_attribute(s,'gene_id'))
    gene_starts = gtf_gene_combined.groupby('gene_id').Start.apply(min)
    gene_ends = gtf_gene_combined.groupby('gene_id').End.apply(max)
    chroms = gtf_gene_combined.groupby('gene_id').Chromosome.apply(lambda s:list(s)[0])
    strands = gtf_gene_combined.groupby('gene_id').Strand.apply(lambda s:list(s)[0])
    
    gtf_dict_stepsize = 10000
    # Create a dictionary for each "bin" of the genome, that maps to a list of genes within or overlapping
    # that bin. The bin size is determined by gtf_dict_stepsize.
    starts_rounded = gene_starts.apply(lambda s:np.floor(s/gtf_dict_stepsize)*gtf_dict_stepsize).values
    ends_rounded = gene_ends.apply(lambda s:np.ceil(s/gtf_dict_stepsize)*gtf_dict_stepsize).values
    gene_ids = gene_starts.index
    start_dict = gene_starts.to_dict()
    end_dict = gene_ends.to_dict()
    gene_dict = defaultdict(list)
    for i in range(len(gene_starts)):
        cur_chrom = chroms[i]
        cur_strand = strands[i]
        cur_start = int(starts_rounded[i])
        cur_end = int(ends_rounded[i])
        cur_gene_id = gene_ids[i]
        for coord in range(cur_start,cur_end+1,gtf_dict_stepsize):
            if not (cur_gene_id in gene_dict[cur_chrom + ':' +  str(coord)]):
                gene_dict[cur_chrom + ':' +  str(coord)+':'+cur_strand].append(cur_gene_id)
                
    # Create a dictionary from genes to exons
    exon_gene_ids = gtf_exon_combined.Attributes.apply(lambda s: get_attribute(s,'gene_id')).values
    exon_starts = gtf_exon_combined.Start.values
    exon_ends = gtf_exon_combined.End.values
    
    exon_gene_start_end_dict = defaultdict(dict)
    for i in range(len(exon_gene_ids)):
        cur_gene_id = exon_gene_ids[i]
        cur_exon_start = exon_starts[i]
        cur_exon_ends = exon_ends[i]
        exon_gene_start_end_dict[cur_gene_id][cur_exon_start] = cur_exon_ends
        
    gene_id_to_gene_names = dict(zip(gtf_gene_combined.Attributes.apply(lambda s: get_attribute(s,'gene_id')),
                                     gtf_gene_combined.Attributes.apply(lambda s: get_attribute(s,'gene_name'))))
    gene_id_to_genome = dict(zip(gtf_gene_combined.Attributes.apply(lambda s: get_attribute(s,'gene_id')),
                                 gtf_gene_combined.Chromosome.apply(lambda s:s.split('_')[0])))
    gene_id_to_strand = dict(zip(gtf_gene_combined.Attributes.apply(lambda s:get_attribute(s,'gene_id')).values,
                                 gtf_gene_combined.Strand.values))
    gene_id_to_chrom = dict(zip(gtf_gene_combined.Attributes.apply(lambda s:get_attribute(s,'gene_id')).values,
                                 gtf_gene_combined.Chromosome.values))
    gene_id_to_biotype = dict(zip(gtf_gene_combined.Attributes.apply(lambda s:get_attribute(s,'gene_id')).values,
                                  gtf_gene_combined.Attributes.apply(lambda s:get_attribute(s,'gene_biotype')).values))
    
    #Save dictionary with gene info
    gene_info = {'gene_bins':gene_dict,
                 'genes_to_exons':exon_gene_start_end_dict,
                 'gene_starts': start_dict,
                 'gene_ends': end_dict,
                 'gene_id_to_name': gene_id_to_gene_names,
                 'gene_id_to_genome':gene_id_to_genome,
                 'gene_id_to_chrom':gene_id_to_chrom,
                 'gene_id_to_strand':gene_id_to_strand,
                 'gene_id_to_biotype':gene_id_to_biotype
                }
    
    with open(output_dir+ '/gene_info.pkl', 'wb') as f:
        pickle.dump(gene_info, f, pickle.HIGHEST_PROTOCOL)

        
def split_attributes(s):
    """ Returns a dictionary from string of attributes in a GTF/GFF file
    """
    att_list = s[:-1].split('; ')
    att_keys = [a.split(' ')[0] for a in att_list]
    att_values = [' '.join(a.split(' ')[1:]) for a in att_list]
    return dict(zip(att_keys,att_values))

def get_attribute(s,att):
    att_value = ''
    try:
        att_value = split_attributes(s)[att].strip('"')
    except:
        att_value = ''
    return att_value
    
def generate_STAR_index(output_dir, nthreads,genomeSAindexNbases,splicing):
    splicing = (splicing=='True')
    if splicing:
        star_command = """STAR  --runMode genomeGenerate\
                                --genomeDir {0}\
                                --genomeFastaFiles {0}/genome.fa\
                                --sjdbGTFfile {0}/exons.gtf\
                                --runThreadN {1}\
                                --limitGenomeGenerateRAM 24000000000\
                                --genomeSAindexNbases {2}""".format(output_dir, nthreads, genomeSAindexNbases)
    else:
        star_command = """STAR  --runMode genomeGenerate --genomeDir {0} --genomeFastaFiles {0}/genome.fa --runThreadN {1} --limitGenomeGenerateRAM 24000000000 --genomeSAindexNbases {2}""".format(output_dir, nthreads, genomeSAindexNbases)
    rc = subprocess.call(star_command, shell=True)
    return rc