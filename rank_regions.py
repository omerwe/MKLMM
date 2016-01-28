import gpUtils
from mklmm import MKLMM
from regionsRanker import RegionsRanker
import scipy.linalg as la
import time
import sys
import numpy as np
np.set_printoptions(precision=4, linewidth=200)



if __name__ == '__main__':

	import argparse
	parser = argparse.ArgumentParser()
	
	#training set test set params
	parser.add_argument('--bfile_train', metavar='bfile_train', default=None, help='Binary plink file with training set')
	parser.add_argument('--bfile', metavar='bfile', default=None, help='Binary plink file with test set (required for removal of PCs)')
	parser.add_argument('--pheno_train', metavar='pheno', default=None, help='Phenotype file of training set, in Plink format')	
	parser.add_argument('--covar_train', metavar='covar_train', default=None, help='covariates file of training set')
	parser.add_argument('--out', metavar='out', default=None, help='output file')
	
	#classifier params
	parser.add_argument('--meanLen', metavar='meanLen',  type=int, default=75000, help='mean region length')
	
	#other params
	parser.add_argument('--extractSim', metavar='extractSim', default=None, help='SNPs subset to use (text file)')
	parser.add_argument('--prev', metavar='prev', type=float, default=0.5, help='Trait prevalence')
	parser.add_argument('--numRemovePCs', metavar='numRemovePCs', type=int, default=0, help='#PCs to remove')	
	parser.add_argument('--reml', metavar='reml', type=int, default=1, help='whether to use REML or not (0 for ML, 1 for REML)')	
	parser.add_argument('--verbose', metavar='verbose', type=int, default=1, help='verbosity level')
	parser.add_argument('--missingPhenotype', metavar='missingPhenotype', default='-9', help='identifier for missing values (default: -9)')
	parser.add_argument('--norm', metavar='norm', default=None, help='SNPs normalization method')
	args = parser.parse_args()

	if (args.bfile_train is None): raise Exception('bfile_train must be supplied')
	if (args.pheno_train is None): raise Exception('pheno_train must be supplied')
	if (args.out is None): raise Exception('output file must be supplied')
		
	#read the input (and standardize it)
	train_bed, train_pheno, covar_train, bed, covar = gpUtils.readInput(args.bfile_train, args.bfile, args.extractSim, args.pheno_train, args.missingPhenotype, args.covar_train, None, args.numRemovePCs, args.norm, args.prev)
	
	
	regionsRanker = RegionsRanker(args.verbose>0)
	regions, betas = regionsRanker.rankRegions(train_bed.val, covar_train, train_pheno, train_bed.pos, args.meanLen, args.reml>0)
	
	#write results to file	
	f = open(args.out, 'w')
	for i in xrange(len(regions)):
		r = regions[i]
		beta = betas[i]
		f.write(str(r[0]) + ' ' + str(r[-1]) + ' ' + '%0.10e'%beta + '\n')
	f.close()
		
