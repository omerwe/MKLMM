import gpUtils
from mklmm import MKLMM
import scipy.linalg as la
import cPickle
import time
import sys
import numpy as np
np.set_printoptions(precision=4, linewidth=200)







def findSignificantRegions(regionsFile, numSNPs, numRegions, regionPercentile=95, verbose=True):
	#read regions file
	chunksArr = np.loadtxt(regionsFile)	
	if (len(chunksArr.shape) == 1): chunksArr = np.column_stack(chunksArr)
	
	regionsArr = chunksArr[:, :2].astype(np.int)
	regionsList = [xrange(regionsArr[i,0], regionsArr[i,1]+1) for i in xrange(regionsArr.shape[0])]
	coeffs = chunksArr[:,2]	
	
	#Join together top regions
	useFilter = np.any(coeffs>0)
	if useFilter: coefThreshold = np.percentile(coeffs[coeffs>0], regionPercentile)
	else: coefThreshold = np.percentile(coeffs, regionPercentile)
	
	if verbose:
		print 'regions percentile threshold:', regionPercentile,
		print '#file regions:', coeffs.shape[0], '#bed regions:', len(regionsList),
		print '#positive coeffs:', np.sum(coeffs>0),
		print '#potential regions:', np.sum(coeffs>coefThreshold),
		print 'coefThreshold: %0.4e'%coefThreshold,
	minCoef = np.min(coeffs)-1
	regionMaxCoef = minCoef		
	coefsAndRegions = []	
	mergedRegionsList = []
	for regionIndex, coef in enumerate(coeffs):			
		if (coef > regionMaxCoef): regionMaxCoef = coef
		if (coef <= coefThreshold or regionIndex == len(coeffs)-1):
			if (regionMaxCoef > coefThreshold): coefsAndRegions.append((regionMaxCoef, mergedRegionsList))
			regionMaxCoef = minCoef
			mergedRegionsList = []
			continue
			
		#if (verbose and len(mergedRegionsList)>0): print 'Merging regions', mergedRegionsList[-1], 'and', regionIndex
		mergedRegionsList.append(regionIndex)
	if verbose: print '#merged regions:', len(coefsAndRegions)
		
		
	#Create top regions
	coefsAndRegions.sort(key = lambda(c, l):c, reverse=True)			
	topRegions = []
	
	#The first kernel is a genome-wide kernel
	regionSNPs = np.ones(numSNPs, dtype=np.bool)
	for region in topRegions: regionSNPs = (regionSNPs & (~region))
	topRegions.append(regionSNPs)
	
	#add additional regions
	for varCompNum in xrange(min(numRegions, len(coefsAndRegions))):
		varCompSNPs = np.zeros(numSNPs, dtype=np.bool)
		for regionNumber in coefsAndRegions[varCompNum][1]:
			if verbose: print 'Adding region', regionNumber, '(with SNPs', str(regionsList[regionNumber][0])+'-'+str(regionsList[regionNumber][-1])+') to kernel', varCompNum+1
			snpsToAdd = regionsList[regionNumber]
			varCompSNPs[list(snpsToAdd)] = True
		topRegions.append(varCompSNPs)

	return topRegions

	
	
	


	
	
if __name__ == '__main__':
	np.random.RandomState(1234567890)
	np.random.seed(1234)

	import argparse
	parser = argparse.ArgumentParser()
	
	#training set test set params
	parser.add_argument('--bfile_train', metavar='bfile_train', default=None, help='Binary plink file with training set')
	parser.add_argument('--pheno_train', metavar='pheno', default=None, help='Phenotype file of training set, in Plink format')	
	parser.add_argument('--covar_train', metavar='covar_train', default=None, help='covariates file of training set')
	parser.add_argument('--train_out', metavar='train_out', default=None, help='output file of training')
	
	#test set params
	parser.add_argument('--train_file', metavar='train_file', default=None, help='Name of file with training data')
	parser.add_argument('--bfile', metavar='bfile', default=None, help='Binary plink file with individuals for which prediction is to be performed')
	parser.add_argument('--covar', metavar='test_covar', default=None, help='covariates file of individuals for which prediction is to be performed')
	parser.add_argument('--out', metavar='out', default=None, help='output file with results of predictions')
	
	#classifier params
	parser.add_argument('--regions', metavar='regions',  default=None, help='regions file')
	parser.add_argument('--kernel', metavar='kernel', default=None, help='kernel type')
	parser.add_argument('--numRegions', metavar='numRegions', type=int, default=None, help='Number of kernel-regions (zero means only a genome-wide kernel)')	
	parser.add_argument('--regionPercentile', metavar='regionPercentile', type=float, default=95, help='percentile above which regions will be merged')
	
	#other params
	parser.add_argument('--extractSim', metavar='extractSim', default=None, help='SNPs subset to use (text file)')
	parser.add_argument('--prev', metavar='prev', type=float, default=0.5, help='Trait prevalence')
	parser.add_argument('--numRemovePCs', metavar='numRemovePCs', type=int, default=0, help='#PCs to remove')	
	parser.add_argument('--reml', metavar='reml', type=int, default=1, help='whether to use REML or not (0 for ML, 1 for REML)')
	parser.add_argument('--maxiter', metavar='maxiter', type=int, default=100, help='max num Optimization iterations')
	parser.add_argument('--verbose', metavar='verbose', type=int, default=1, help='verbosity level')
	parser.add_argument('--missingPhenotype', metavar='missingPhenotype', default='-9', help='identifier for missing values (default: -9)')
	parser.add_argument('--norm', metavar='norm', default=None, help='SNPs normalization method')	
	args = parser.parse_args()

	if (args.bfile_train is None): raise Exception('bfile_train must be supplied')
	if (args.regions is None): raise Exception('regions file must be supplied')
	if (args.kernel is None and (args.bfile is None or args.train_file is None)): raise Exception('kernel type must be supplied')
	if (args.numRegions is None): raise Exception('numRegions must be supplied')
	if (args.pheno_train is None): raise Exception('pheno_train must be supplied')
	
	if (args.bfile is None and args.train_out is None): raise Exception('Either bfile or train_out must be supplied')	
	if (args.train_file is not None and args.train_out is not None): raise Exception('train_file and train_out cannot both be supplied')	
	if (args.bfile is not None and args.out is None): raise Exception('An output file name must be supplied with bfile')
	if (args.covar_train is None and args.covar is not None): raise Exception('covar cannot be supplied without covar_train')
	if (args.bfile is None and args.out is not None): raise Exception('output file cannot be created without a bfile')
	if (args.numRemovePCs>0 and args.bfile is None): raise Exception('Cannot remove PCs without a testing set')
	
	#load training dictionary
	if (args.train_file is None): trainDict = dict([])
	else:		
		f = open(args.train_file, 'rb')
		trainDict = cPickle.load(f)
		f.close()
		
	#read the input (and standardize it)
	train_bed, train_pheno, covar_train, bed, covar = gpUtils.readInput(args.bfile_train, args.bfile, args.extractSim, args.pheno_train, args.missingPhenotype, args.covar_train, args.covar, args.numRemovePCs, args.norm, args.prev)
	np.random.RandomState(1234567890)
	np.random.seed(1234)
		
	#read regions file
	regions = findSignificantRegions(args.regions, train_bed.val.shape[1], args.numRegions, args.regionPercentile, args.verbose>0)		
	
	#Create an MKLMM object	
	mklmm = MKLMM(args.verbose>0)
	
	#Train the MKLMM	
	if (args.train_file is None):
		trainDict2 = mklmm.fit(train_bed.val, covar_train, train_pheno, regions, args.kernel, args.reml>0, args.maxiter)		
		trainDict.update(trainDict2)
		
		#save training dictionary
		if (args.train_out is not None):
			f = open(args.train_out, 'wb')
			cPickle.dump(trainDict, f, protocol=2)
			f.close()
			
	#perform prediction
	if (args.bfile is not None):
		mu, var = mklmm.predict(train_bed.val, covar_train, train_pheno, regions, bed.val, covar, trainDict)
		
		#print predictions
		f = open(args.out, 'w')
		for i in xrange(bed.iid.shape[0]):
			f.write(str(bed.iid[i,0]) + ' ' + str(bed.iid[i,1]) + ' ' +str(mu[i]) + ' ' + str(var[i]) + '\n')
		f.close()
		
	
	





