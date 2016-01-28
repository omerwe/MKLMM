import numpy as np
from optparse import OptionParser
import scipy.linalg as la
import scipy.stats as stats
import scipy.linalg.blas as blas
import scipy.optimize as optimize
import pandas as pd
import csv
import time
from pysnptools.snpreader.bed import Bed
import pysnptools.util as pstutil
import pysnptools.util.pheno as phenoUtils
np.set_printoptions(precision=4, linewidth=200)



def loadData(bfile, extractSim, phenoFile, missingPhenotype='-9', loadSNPs=False, standardize=True):
	bed = Bed(bfile)
	
	if (extractSim is not None):
		f = open(extractSim)
		csvReader = csv.reader(f)
		extractSnpsSet = set([])
		for l in csvReader: extractSnpsSet.add(l[0])			
		f.close()		
		keepSnpsInds = [i for i in xrange(bed.sid.shape[0]) if bed.sid[i] in extractSnpsSet]		
		bed = bed[:, keepSnpsInds]
		
	phe = None
	if (phenoFile is not None):	bed, phe = loadPheno(bed, phenoFile, missingPhenotype)
	
	if (loadSNPs):
		bed = bed.read()
		if (standardize): bed = bed.standardize()	
	
	return bed, phe
	
	
def loadPheno(bed, phenoFile, missingPhenotype='-9', keepDict=False):
	pheno = phenoUtils.loadOnePhen(phenoFile, missing=missingPhenotype, vectorize=True)
	checkIntersection(bed, pheno, 'phenotypes')
	bed, pheno = pstutil.intersect_apply([bed, pheno])
	if (not keepDict): pheno = pheno['vals']
	return bed, pheno
	
	
def checkIntersection(bed, fileDict, fileStr, checkSuperSet=False):
	bedSet = set((b[0], b[1]) for b in bed.iid)
	fileSet = set((b[0], b[1]) for b in fileDict['iid'])
	
	if checkSuperSet:
		if (not fileSet.issuperset(bedSet)): raise Exception(fileStr + " file does not include all individuals in the bfile")
	
	intersectSet = bedSet.intersection(fileSet)
	if (len(intersectSet) != len (bedSet)):
		print len(intersectSet), 'individuals appear in both the plink file and the', fileStr, 'file'


def loadCovars(bed, covarFile):
	covarsDict = phenoUtils.loadPhen(covarFile)
	checkIntersection(bed, covarsDict, 'covariates', checkSuperSet=True)
	_, covarsDict = pstutil.intersect_apply([bed, covarsDict])
	covar = covarsDict['vals']
	return covar	
	
	
def getExcludedChromosome(bfile, chrom):
	bed = Bed(bfile)	
	indsToKeep = (bed.pos[:,0] != chrom)
	bed = bed[:, indsToKeep]	
	return bed.read().standardize()
	
def getChromosome(bfile, chrom):
	bed = Bed(bfile)
	indsToKeep = (bed.pos[:,0] == chrom)
	bed = bed[:, indsToKeep]	
	return bed.read().standardize()
	

def _fixupBedAndPheno(bed, pheno, missingPhenotype='-9'):
	bed = _fixupBed(bed)
	bed, pheno = _fixup_pheno(pheno, bed, missingPhenotype)
	return bed, pheno
	
def _fixupBed(bed):
	if isinstance(bed, str):
		return Bed(bed).read().standardize()
	else: return bed

def _fixup_pheno(pheno, bed=None, missingPhenotype='-9'):
	if (isinstance(pheno, str)):
		if (bed is not None):
			bed, pheno = loadPheno(bed, pheno, missingPhenotype, keepDict=True)
			return bed, pheno
		else:
			phenoDict = phenoUtils.loadOnePhen(pheno, missing=missingPhenotype, vectorize=True)
			return phenoDict
	else:
		if (bed is not None): return bed, pheno			
		else: return pheno
		
		
#Mean-impute missing SNPs	
def imputeSNPs(X):
	snpsMean = np.nanmean(X, axis=0)
	isNan = np.isnan(X)
	for i,m in enumerate(snpsMean): X[isNan[:,i], i] = m
		
	return X
	
	

def readInput(bfile_train, bfile, extractSim, pheno_train, missingPhenotype, covar_train, covar, numRemovePCs, norm, prev):
	
	#Read bfiles and impute training SNPs
	numTrain = 0
	train_bed, train_pheno = loadData(bfile_train, extractSim, pheno_train, missingPhenotype, loadSNPs=True, standardize=False)		
	numTrain = train_bed.iid.shape[0]
	snpsMean = np.nanmean(train_bed.val, axis=0)
	snpsStd = np.nanstd(train_bed.val, axis=0)
	isNan = np.isnan(train_bed.val)
	for i,m in enumerate(snpsMean): train_bed.val[isNan[:,i], i] = m
		
	#impute test SNPs
	numTest = 0
	if (bfile is None): bed = None
	else:
		bed, _ = loadData(bfile, extractSim, None, missingPhenotype, loadSNPs=True, standardize=False)
		if (bed.val.shape[1] != train_bed.val.shape[1]): raise Exception('bfile_train and bfile have a different number of SNPs')
		numTest = bed.iid.shape[0]
		isNan = np.isnan(bed.val)
		for i,m in enumerate(snpsMean): bed.val[isNan[:,i], i] = m

	#load training covariates and mean-impute
	if (covar_train is None): covar_train = np.empty((numTrain, 0))
	else:
		covar_train = loadCovars(train_bed, covar_train)				
		covMean = np.nanmean(covar_train, axis=0)
		covStd = np.nanstd(covar_train, axis=0)
		isNan = np.isnan(covar_train)
		for i,m in enumerate(covMean): covar_train[isNan[:,i], i] = m		
		covar_train -= covMean
		covar_train /= covStd		
	
			
	#load test covariates and mean-impute
	if (covar is None): covar_test = np.empty((numTest, 0))
	else:
		covar_test = loadCovars(bed, covar)
		if (covar_test.shape[1] != covar_train.shape[1]): raise Exception('covar_train and covar have different number of covariates')
		isNan = np.isnan(covar_test)
		for i,m in enumerate(covMean): covar_test[isNan[:,i], i] = m	
		covar_test -= covMean
		covar_test /= covStd
	
	covar_train = np.concatenate((np.ones((numTrain, 1)), covar_train), axis=1)				#add a column of ones
	covar_test = np.concatenate((np.ones((numTest, 1)), covar_test), axis=1)				#add a column of ones
		
		
	#Remove top PCs
	if (numRemovePCs>0):
		print 'Removing', numRemovePCs, 'top PCs...'
		X = np.concatenate((train_bed.val, bed.val), axis=0)
		X = removeTopPCs(X, numRemovePCs)
		train_bed.val = X[:numTrain, :]
		bed.val = X[numTrain:, :]
		print 'done'
		
	#standardize SNPs	
	snpsMean, snpsStd = normalizeSNPs(norm, train_bed.val, train_pheno, prev)
	train_bed.val -= snpsMean
	train_bed.val /= snpsStd
		
	if (bfile is not None):
		bed.val -= snpsMean
		bed.val /= snpsStd
		
		
	#standardize phenotype if it's binary
	isBinaryPhenotype = (len(np.unique(train_pheno)) == 2)
	if isBinaryPhenotype:
		pheMean = train_pheno.mean()
		train_pheno[train_pheno <= pheMean] = 0
		train_pheno[train_pheno > pheMean] = 1
	#otherwise, normalize it to have variance 1.0
	else:
		train_pheno /= train_pheno.std()
		

	
	return train_bed, train_pheno, covar_train, bed, covar_test
	
	
#regress out top PCs
def removeTopPCs(X, numRemovePCs):	
	t0 = time.time()
	X_mean = X.mean(axis=0)
	X -= X_mean
	XXT = symmetrize(blas.dsyrk(1.0, X, lower=0))
	s,U = la.eigh(XXT)
	if (np.min(s) < -1e-4): raise Exception('Negative eigenvalues found')
	s[s<0]=0
	ind = np.argsort(s)[::-1]
	U = U[:, ind]
	s = s[ind]
	s = np.sqrt(s)
		
	#remove null PCs
	ind = (s>1e-6)
	U = U[:, ind]
	s = s[ind]
	
	V = X.T.dot(U/s)	
	#print 'max diff:', np.max(((U*s).dot(V.T) - X)**2)
	X = (U[:, numRemovePCs:]*s[numRemovePCs:]).dot((V.T)[numRemovePCs:, :])
	X += X_mean
	
	return X
	

def normalizeSNPs(normMethod, X, y, prev=None, frqFile=None):
	if (normMethod == 'frq'):
		print 'flipping SNPs for standardization...'
		empMean = X.mean(axis=0) / 2.0
		X[:, empMean>0.5] = 2 - X[:, empMean>0.5]		
		mafs = np.loadtxt(frqFile, usecols=[1,2]).mean(axis=1)
		snpsMean = 2*mafs
		snpsStd = np.sqrt(2*mafs*(1-mafs))
	elif (normMethod == 'controls'):
		controls = (y<y.mean())
		cases = ~controls
		snpsMeanControls, snpsStdControls = X[controls, :].mean(axis=0), X[controls, :].std(axis=0)
		snpsMeanCases, snpsStdCases = X[cases, :].mean(axis=0), X[cases, :].std(axis=0)
		snpsMean = (1-prev)*snpsMeanControls + prev*snpsMeanCases
		snpsStd = (1-prev)*snpsStdControls + prev*snpsStdCases
	elif (normMethod is None): snpsMean, snpsStd = X.mean(axis=0), X.std(axis=0)
	else: raise Exception('Unrecognized normalization method: ' + normMethod)
	
	return snpsMean, snpsStd
	
	
def symmetrize(X): return X + X.T - np.diag(X.diagonal())
	

#this code is directly translated from the GPML Matlab package (see attached license)
def minimize(X, f, length, *varargin):

	realmin = np.finfo(np.double).tiny 
	INT = 0.1    #don't reevaluate within 0.1 of the limit of the current bracket
	EXT = 3.0    #extrapolate maximum 3 times the current step-size
	MAX = 20     #max 20 function evaluations per line search
	RATIO = 10   #maximum allowed slope ratio
	SIG = 0.1  
	RHO = SIG/2 #SIG and RHO are the constants controlling the Wolfe-
	#Powell conditions. SIG is the maximum allowed absolute ratio between
	#previous and new slopes (derivatives in the search direction), thus setting
	#SIG to low (positive) values forces higher precision in the line-searches.
	#RHO is the minimum allowed fraction of the expected (from the slope at the
	#initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
	#Tuning of SIG (depending on the nature of the function to be optimized) may
	#speed up the minimization; it is probably not worth playing much with RHO.
		
		
	#The code falls naturally into 3 parts, after the initial line search is
	#started in the direction of steepest descent. 1) we first enter a while loop
	#which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
	#have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
	#enter the second loop which takes p2, p3 and p4 chooses the subinterval
	#containing a (local) minimum, and interpolates it, unil an acceptable point
	#is found (Wolfe-Powell conditions). Note, that points are always maintained
	#in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
	#conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
	#was a problem in the previous line-search. Return the best value so far, if
	#two consecutive line-searches fail, or whenever we run out of function
	#evaluations or line-searches. During extrapolation, the "f" function may fail
	#either with an error or returning Nan or Inf, and minimize should handle this
	#gracefully.
	
	
	red=1.0	
	if length>0: S = 'Linesearch'
	else: S = 'Function evaluation'

	funcalls = 0
	i = 0                              #zero the run length counter
	ls_failed = False                  #no previous line search has failed	
	
	f0, df0 = f(X, *varargin)          #get function value and gradient		
	funcalls+=1
	
	#print S, 'iteration', i, 'Value: %4.6e'%f0
	fX = [f0]
	if (length<0): i+=1 	#count epochs?!	
	s = -df0
	d0 = -s.dot(s)        #initial search direction (steepest) and slope
	x3 = red/(1-d0)       #initial step is red/(|s|+1)
	
	while (i < np.abs(length)):
		if (length>0): i+=1 	#count epochs?!	
		X0 = X.copy()
		F0 = f0
		dF0 = df0.copy()
		M = (MAX if (length>0) else np.minimum(MAX, -length-i))
		while True:
			x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0.copy()
			success = False
			while (not success and M>0):			
				try:
					M-=1
					if (length<0): i+=1
					f3, df3 = f(X+x3*s, *varargin)
					funcalls+=1
					if (np.isnan(f3) or np.isinf(f3)): raise Exception('')
					success=True
				except:
					x3 = (x2+x3)/2.0		#bisect and try again
		
			if (f3 < F0):					#keep best values
				X0 = X+x3*s
				F0 = f3
				dF0 = df3.copy()
			d3 = df3.dot(s)       			#new slope
			if (d3 > SIG*d0 or f3 > f0+x3*RHO*d0 or M == 0): break  #are we done extrapolating?

	
			x1 = x2; f1 = f2; d1 = d2;                       # move point 2 to point 1
			x2 = x3; f2 = f3; d2 = d3;                       # move point 3 to point 2
			A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 # make cubic extrapolation
			B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
			x3 = x1-d1*(x2-x1)**2 / (B+np.sqrt(B*B-A*d1*(x2-x1))) # num. error possible, ok!
			if (not np.isreal(x3) or np.isnan(x3) or np.isinf(x3) or x3 < 0): x3 = x2*EXT # num prob | wrong sign?				               
			elif (x3 > x2*EXT): x3 = x2*EXT     			# new point beyond extrapolation limit?	 extrapolate maximum amount
			elif (x3 < x2+INT*(x2-x1)): x3 = x2+INT*(x2-x1) # new point too close to previous point?
				
		while ((np.abs(d3) > -SIG*d0 or f3 > f0+x3*RHO*d0) and M > 0):  # keep interpolating
			if (d3 > 0 or f3 > f0+x3*RHO*d0):                	# choose subinterval
				x4 = x3; f4 = f3; d4 = d3;                      # move point 3 to point 4
			else:
				x2 = x3; f2 = f3; d2 = d3;                      # move point 3 to point 2			
			if (f4 > f0):           
				x3 = x2-(0.5*d2*(x4-x2)**2)/(f4-f2-d2*(x4-x2))  # quadratic interpolation
			else:
				A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                # cubic interpolation
				B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
				x3 = x2+(np.sqrt(B*B-A*d2*(x4-x2)**2)-B)/A;     # num. error possible, ok!
			if (np.isnan(x3) or np.isinf(x3)):
				x3 = (x2+x4)/2;               					# if we had a numerical problem then bisect			
			x3 = np.maximum(np.minimum(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2));  # don't accept too close
			
			
			f3, df3 = f(X+x3*s, *varargin)
			funcalls+=1
			if (f3 < F0):										 # keep best values
				X0 = X+x3*s; F0 = f3; dF0 = df3.copy() 
			M-=1
			if (length<0): i+=1 								 # count epochs?!
			d3 = df3.dot(s)          							 # new slope
			
		if (np.abs(d3) < -SIG*d0 and f3 < f0+x3*RHO*d0):         		 # if line search succeeded
			X = X+x3*s; f0 = f3
			fX.append(f0)                     							 # update variables
			#print S, i, 'Value: %4.6e'%f0
			s = (df3.dot(df3)-df0.dot(df3)) / (df0.dot(df0)) * s - df3   # Polack-Ribiere CG direction
			df0 = df3.copy()                                             # swap derivatives
			d3 = d0; d0 = df0.dot(s)
			if (d0 > 0):                                 # new slope must be negative
				s = -df0
				d0 = -s.dot(s)                  		 # otherwise use steepest direction
			x3 *= np.minimum(RATIO, d3/(d0-realmin))     # slope ratio but max RATIO
			ls_failed = False                            # this line search did not fail
		else:
			X = X0; f0 = F0; df0 = dF0.copy()            # restore best point so far
			if (ls_failed or i > np.abs(length)):        # line search failed twice in a row
				break                             		 # or we ran out of time, so we give up
			s = -df0; d0 = -s.dot(s);                    # try steepest
			x3 = 1.0/(1.0-d0)                     
			ls_failed = True                             # this line search failed
			
	#print S, 'iteration', i, 'Value: %4.6e'%f0
	
	return optimize.OptimizeResult(fun=F0, x=X0, nit=i, nfev=funcalls, success=True, status=0, message='')
