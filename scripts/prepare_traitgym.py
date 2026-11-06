'''
Download a TraitGym dataset (https://huggingface.co/datasets/songlab/TraitGym) and convert it to the
bed format used by predict_variant_effects.py.

TraitGym gives 1-based positions on GRCh38, while predict_variant_effects.py reads 0-based coordinates.
Writing start == end gives windows of exactly 2 * extra_context bp, with the variant at index extra_context.
'''
import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from huggingface_hub import hf_hub_download


def main():

    parser = argparse.ArgumentParser('Convert a TraitGym dataset to a bed file')
    parser.add_argument('dataset', choices=['mendelian_traits_matched_9', 'complex_traits_matched_9', 'mendelian_traits_all', 'complex_traits_all'], type=str, help='TraitGym dataset')
    parser.add_argument('out_file', type=str, help='Path to the output bed file')
    parser.add_argument('genome', type=str, help='Path to the GRCh38 reference genome fasta file')
    parser.add_argument('--chromosomes', type=str, nargs='+', default=None, help='Keep only these chromosomes, e.g. 19 for a quick test')

    args = parser.parse_args()

    df = pd.read_parquet(hf_hub_download('songlab/TraitGym', f'{args.dataset}/test.parquet', repo_type='dataset'))
    if args.chromosomes is not None:
        df = df[df['chrom'].isin(args.chromosomes)].reset_index(drop=True)

    genome = SeqIO.to_dict(SeqIO.parse(args.genome, 'fasta'))
    prefix = 'chr' if any(name.startswith('chr') for name in genome) else ''

    bed = df.drop(columns=['chrom', 'pos', 'ref', 'alt', 'label'])
    bed.insert(0, 'chromosome', prefix + df['chrom'])
    bed.insert(1, 'start', df['pos'] - 1)
    bed.insert(2, 'end', df['pos'] - 1)
    bed.insert(3, 'ref', df['ref'])
    bed.insert(4, 'alt', df['alt'])
    bed.insert(5, 'label', df['label'].astype(int))

    # a wrong assembly or coordinate convention would not raise any error later on, it would only give wrong embeddings
    missing = sorted(set(bed['chromosome']) - set(genome))
    if missing:
        raise ValueError(f'Chromosomes not found in {args.genome}: {missing}')
    wrong = sum(str(genome[c].seq[s]).upper() != r for c, s, r in zip(bed['chromosome'], bed['start'], bed['ref']))
    if wrong > 0:
        raise ValueError(f'{wrong} of {len(bed)} reference alleles do not match {args.genome}. Is it GRCh38?')

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    bed.to_csv(args.out_file, sep='\t', index=False)
    print(f'{len(bed)} variants ({bed["label"].sum()} positive) written to {args.out_file}')


if __name__ == '__main__':
    main()
