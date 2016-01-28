import numpy as np
import scipy.linalg as la
import time
import sys
import scipy.optimize as optimize
np.set_printoptions(precision=4, linewidth=200)

class RegionsRanker:

	def __init__(self, verbose=False):
		self.verbose = verbose
		pass
		
		
	def createRegionsList(self, pos, regionLength):
		pnt1 = 0
		pnt2 = 1
		halfLength = regionLength/2.0
		regions = []
		while (pnt1 < pos.shape[0]-1):
			while (pnt2 < (pos.shape[0]-2) and (pos[pnt2+1,2]-pos[pnt1,2] < regionLength) and pos[pnt2+1,0]==pos[pnt1,0]): pnt2+=1			
			regions.append(xrange(pnt1, pnt2+1))
			oldPnt1 = pnt1
			pnt1+=1
			while (pnt1 < (pos.shape[0]-2) and (pos[pnt1+1,2]-pos[oldPnt1,2] < halfLength) and pos[pnt1+1,0]==pos[oldPnt1,0]): pnt1+=1
			if (pnt1 >= pnt2): pnt2+=1
			if (pnt2 > (pos.shape[0]-2)):
				regions.append(xrange(pnt1, pos.shape[0]))
				break
			if (pos[pnt2,0] != pos[pnt1,0]): pnt1=pnt2
		return regions
			


	def rankRegions(self, X, C, y, pos, regionLength, reml=True):
	
		#get resiong list
		regionsList = self.createRegionsList(pos, regionLength)
			
		#precompute log determinant of covariates
		XX = C.T.dot(C)
		[Sxx,Uxx]= la.eigh(XX)
		logdetXX  = np.log(Sxx).sum()
		
		#score each region
		betas = np.zeros(len(regionsList))
		for r_i, r in enumerate(regionsList):
			regionSize = len(r)
		
			if (self.verbose and r_i % 1000==0):
				print 'Testing region ' + str(r_i+1)+'/'+str(len(regionsList)),
				print 'with', regionSize, 'SNPs\t'
			
			s,U = self.eigenDecompose(X[:, np.array(r)], None)
			sig2g_kernel, sig2e_kernel, fixedEffects, ll = self.optSigma2(U, s, y, C, logdetXX, reml)
			betas[r_i] = ll
		
		return regionsList, betas
		
		
	### this code is taken from the FastLMM package (see attached license)###
	def lleval(self, Uy, UX, Sd, yKy, logdetK, logdetXX, logdelta, UUXUUX, UUXUUy, UUyUUy, numIndividuals, reml):
		N = numIndividuals
		D = UX.shape[1]
				
		UXS = UX / np.lib.stride_tricks.as_strided(Sd, (Sd.size, D), (Sd.itemsize,0))
		XKy = UXS.T.dot(Uy)			
		XKX = UXS.T.dot(UX)	
		
		if (Sd.shape[0] < numIndividuals):
			delta = np.exp(logdelta)
			denom = delta
			XKX += UUXUUX / denom
			XKy += UUXUUy / denom
			yKy += UUyUUy / denom			
			logdetK += (numIndividuals-Sd.shape[0]) * logdelta		
			
		[SxKx,UxKx]= la.eigh(XKX)	
		i_pos = SxKx>1E-10
		beta = np.dot(UxKx[:,i_pos], (np.dot(UxKx[:,i_pos].T, XKy) / SxKx[i_pos]))
		r2 = yKy-XKy.dot(beta)

		if reml:
			logdetXKX = np.log(SxKx).sum()
			sigma2 = (r2 / (N - D))
			ll =  -0.5 * (logdetK + (N-D)*np.log(2.0*np.pi*sigma2) + (N-D) + logdetXKX - logdetXX)
		else:
			sigma2 = r2 / N
			ll =  -0.5 * (logdetK + N*np.log(2.0*np.pi*sigma2) + N)
			
		return ll, sigma2, beta, r2
		
		
		
	def negLLevalLong(self, logdelta, s, Uy, UX, logdetXX, UUXUUX, UUXUUy, UUyUUy, numIndividuals, reml, returnAllParams=False):
		Sd = s + np.exp(logdelta)
		UyS = Uy / Sd
		yKy = UyS.T.dot(Uy)
		logdetK = np.log(Sd).sum()
		null_ll, sigma2, beta, r2 = self.lleval(Uy, UX, Sd, yKy, logdetK, logdetXX, logdelta, UUXUUX, UUXUUy, UUyUUy, numIndividuals, reml)
		if returnAllParams: return null_ll, sigma2, beta, r2
		else: return -null_ll
		
		
	def optSigma2(self, U, s, y, covars, logdetXX, reml, ldeltamin=-5, ldeltamax=5):

		#Prepare required matrices
		Uy = U.T.dot(y).flatten()
		UX = U.T.dot(covars)		
		
		if (U.shape[1] < U.shape[0]):
			UUX = covars - U.dot(UX)
			UUy = y - U.dot(Uy)
			UUXUUX = UUX.T.dot(UUX)
			UUXUUy = UUX.T.dot(UUy)
			UUyUUy = UUy.T.dot(UUy)
		else: UUXUUX, UUXUUy, UUyUUy = None, None, None
		numIndividuals = U.shape[0]
		ldeltaopt_glob = optimize.minimize_scalar(self.negLLevalLong, bounds=(-5, 5), method='Bounded', args=(s, Uy, UX, logdetXX, UUXUUX, UUXUUy, UUyUUy, numIndividuals, reml)).x
		
		ll, sig2g, beta, r2 = self.negLLevalLong(ldeltaopt_glob, s, Uy, UX, logdetXX, UUXUUX, UUXUUy, UUyUUy, numIndividuals, reml, returnAllParams=True)
		sig2e = np.exp(ldeltaopt_glob) * sig2g
			
		return sig2g, sig2e, beta, ll
		
		
	def eigenDecompose(self, X, K, normalize=True):
		if (X.shape[1] >= X.shape[0]):
			s,U = la.eigh(K)
		else:
			U, s, _ = la.svd(X, check_finite=False, full_matrices=False)
			if (s.shape[0] < U.shape[1]): s = np.concatenate((s, np.zeros(U.shape[1]-s.shape[0])))	#note: can use low-rank formulas here			
			s=s**2
			if normalize: s /= float(X.shape[1])
		if (np.min(s) < -1e-10): raise Exception('Negative eigenvalues found')
		s[s<0]=0	
		ind = np.argsort(s)[::-1]
		U = U[:, ind]
		s = s[ind]	
		
		return s,U
		
		
	