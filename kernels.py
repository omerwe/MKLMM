import numpy as np
import gpUtils
import scipy.linalg.blas as blas
import sys



#Many parts of this code are based on the Matlab GPML package (see attached license)


def raiseK(K, d):
	if (d<1): raise Exception('non-positive power requested!')
	if (d==1): return K	
	Kpower = K**2
	for i in xrange(d-2): Kpower*=K				#this is faster than direct power
	return Kpower



def sq_dist(a, b=None):
	#mean-center for numerical stability
	D, n = a.shape[0], a.shape[1]
	if (b is None):
		mu = a.mean(axis=1)
		a -= mu[:, np.newaxis]
		b = a
		m = n
		aSq = np.sum(a**2, axis=0)
		bSq = aSq
	else:
		d, m = b.shape[0], b.shape[1]
		if (d != D): raise Exception('column lengths must agree')
		mu = (float(m)/float(m+n))*b.mean(axis=1) + (float(n)/float(m+n))*a.mean(axis=1)
		a -= mu[:, np.newaxis]
		b -= mu[:, np.newaxis]		
		aSq = np.sum(a**2, axis=0)
		bSq = np.sum(b**2, axis=0)
		
	C = np.tile(np.column_stack(aSq).T, (1, m)) + np.tile(bSq, (n, 1)) - 2*a.T.dot(b)
	C = np.maximum(C, 0)	#remove numerical noise
	return C
	
#evaluate 'power sums' of the individual terms in Z
def elsympol(Z,R):
	sz = Z.shape	
	E = np.zeros((sz[0], sz[1], R+1))		#E(:,:,r+1) yields polynomial r
	E[:,:,0] = np.ones(sz[:2])
	if (R==0): return E  #init recursion
	
	P = np.zeros((sz[0], sz[1], R))
	for r in xrange(1,R+1): P[:,:,r-1] = np.sum(Z**r, axis=2)
	E[:,:,1] = P[:,:,0]
	if (R==1): return E  #init recursion
	
	for r in xrange(2,R+1):
		for i in xrange(1,r+1):
			E[:,:,r] += P[:,:,i-1]*E[:,:,r-i] * (-1)**(i-1)/float(r)
	return E
		
	
	


class Kernel:
	
	def __init__(self):
		self.epsilon = 1e-80
		self.prevParams = None
		self.cache = dict([])
		
	def checkParams(self, params):
		if (len(params.shape) != 1 or params.shape[0] != self.getNumParams()): raise Exception('Incorrect number of parameters')
		
	def checkParamsI(self, params, i):
		self.checkParams(params)
		if (i<0 or i>=params.shape[0]): raise Exception('Invalid parameter number')
		
	def sameParams(self, params, i=None):
		if (self.prevParams is None): return False
		if (i is None): return (np.max(np.abs(params-self.prevParams)) < self.epsilon)
		return ((np.abs(params[i]-self.prevParams[i])) < self.epsilon)
		
	def saveParams(self, params): self.prevParams = params.copy()
	def getNumParams(self): raise Exception('getNumParams() called from base Kernel class')	
	def getTrainKernel(self, params): raise Exception('getTrainKernel() called from base Kernel class')	
	def getTrainTestKernel(self, params, Xtest): raise Exception('getTrainTestKernel() called from base Kernel class')
	def getTestKernelDiag(self, params, Xtest): raise Exception('getTestKernelDiag() called from base Kernel class')
	def deriveKernel(self, params, i): raise Exception('deriveKernel() called from base Kernel class')
	
	
class IdentityKernel(Kernel):
	
	def __init__(self, n):
		Kernel.__init__(self)
		self.n = n
			
	def getNumParams(self): return 0
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		return np.eye(self.n)
		
	def deriveKernel(self, params, i): raise Exception('Identity Kernel cannot be derived')
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		return np.zeros((self.n, Xtest.shape[0]))
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		return np.ones(Xtest.shape[0])

	

class linearKernel(Kernel):
	
	def __init__(self, X):
		Kernel.__init__(self)		
		self.X_scaled = X / X.shape[1]		
		if (X.shape[1] >= X.shape[0]): self.XXT = gpUtils.symmetrize(blas.dsyrk(1.0/X.shape[1], X, lower=0))
		else:
			self.XXT = None
			self.X = X
			
	def getNumParams(self): return 0
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.XXT is not None): return self.XXT
		else: return self.X_scaled.dot(self.X.T)
		
	def deriveKernel(self, params, i): raise Exception('Linear Kernel cannot be derived')
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		return self.X_scaled.dot(Xtest.T)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		return np.sum(Xtest**2, axis=1) / Xtest.shape[1]
		
	
	

class ScaledKernel(Kernel):

	def __init__(self, kernel):
		Kernel.__init__(self)
		self.kernel = kernel

	def getNumParams(self): return 1+self.kernel.getNumParams()
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']
		
		coeff = np.exp(2*params[-1])
		K = coeff * self.kernel.getTrainKernel(params[:-1])
		self.cache['getTrainKernel'] = K
		self.saveParams(params)
		return K		
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		coeff = np.exp(2*params[-1])		
		if (i==params.shape[0]-1): K = 2*self.getTrainKernel(params)
		else: K = coeff * self.kernel.deriveKernel(params[:-1], i)
		return K
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		coeff = np.exp(2*params[-1])
		return coeff * self.kernel.getTrainTestKernel(params[:-1], Xtest)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		coeff = np.exp(2*params[-1])
		return coeff * self.kernel.getTestKernelDiag(params[:-1], Xtest)
		
			
		
		
class SumKernel(Kernel):

	def __init__(self, kernels):
		Kernel.__init__(self)
		self.kernels = kernels

	def getNumParams(self): return np.sum([k.getNumParams() for k in self.kernels])
	
	def getTrainKernel(self, params, i=None):
		self.checkParams(params)		
		K = 0
		params_ind = 0
		for k_i, k in enumerate(self.kernels):
			numHyp = k.getNumParams()
			currK = k.getTrainKernel(params[params_ind:params_ind+numHyp])			
			if (i is None): K += currK
			elif (i==k_i): return currK
			params_ind += numHyp
		return K
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		params_ind = 0
		for k in self.kernels:
			numHyp = k.getNumParams()
			if (i not in xrange(params_ind, params_ind+numHyp)):
				params_ind += numHyp
				continue			
			return k.deriveKernel(params[params_ind:params_ind+numHyp], i-params_ind)
		raise Exception('invalid parameter index')
		
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		if (len(Xtest) != len(self.kernels)): raise Exception('Xtest should be a list with length equal to #kernels!')
		K = 0
		params_ind = 0
		for k_i, k in enumerate(self.kernels):
			numHyp = k.getNumParams()
			K += k.getTrainTestKernel(params[params_ind:params_ind+numHyp], Xtest[k_i])
			params_ind += numHyp
		return K
		
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		if (len(Xtest) != len(self.kernels)): raise Exception('Xtest should be a list with length equal to #kernels!')
		diag = 0
		params_ind = 0
		for k_i, k in enumerate(self.kernels):
			numHyp = k.getNumParams()
			diag += k.getTestKernelDiag(params[params_ind:params_ind+numHyp], Xtest[k_i])
			params_ind += numHyp
		return diag
		
		
			
		
#the only parameter here is the bias term c...
class PolyKernel(Kernel):
	def __init__(self, kernel):
		Kernel.__init__(self)
		self.kernel = kernel

	def getNumParams(self): return 1+self.kernel.getNumParams()
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		c = np.exp(params[-1])
		K = self.kernel.getTrainKernel(params[:-1]) + c
		Kpower = K**2												#this is very fast, so we do it manually
		for i in xrange(self.polyDegree-2): Kpower*=K				#this is faster than direct power
		return Kpower
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		c = np.exp(params[-1])
		
		#K = self.kernel.getTrainKernel(params[:-1])
		#deriv1 = self.polyDegree * (c+K)**(self.polyDegree-1)
		
		K = self.kernel.getTrainKernel(params[:-1]) + c		
		if (self.polyDegree==2): Kpower=K
		else:
			Kpower=K**2
			for i in xrange(self.polyDegree-3): Kpower*=K				#this is faster than direct power
		deriv1 = self.polyDegree * Kpower
		
		if (i==params.shape[0]-1): K_deriv = deriv1*c
		else: 	   K_deriv = deriv1 * self.kernel.deriveKernel(params[:-1], i)
		return K_deriv
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		c = np.exp(params[-1])
		return (self.kernel.getTrainTestKernel(params[:-1], Xtest) + c)**self.polyDegree
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		c = np.exp(params[-1])		
		return (self.kernel.getTestKernelDiag(params[:-1], Xtest) + c)**self.polyDegree
		

class Poly2Kernel(PolyKernel):
	def __init__(self, kernel):
		PolyKernel.__init__(self, kernel)
		self.polyDegree = 2

class Poly3Kernel(PolyKernel):
	def __init__(self, kernel):
		PolyKernel.__init__(self, kernel)
		self.polyDegree = 3
		
		
		
		
		
		
class PolyKernelHomo(Kernel):
	def __init__(self, kernel):
		Kernel.__init__(self)
		self.kernel = kernel

	def getNumParams(self): return self.kernel.getNumParams()
	
	def getTrainKernel(self, params):
		self.checkParams(params)		
		K = self.kernel.getTrainKernel(params)
		return raiseK(K, self.polyDegree)
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		K = self.kernel.getTrainKernel(params)
		Kpower = raiseK(K, self.polyDegree-1)
		deriv1 = self.polyDegree * Kpower
		return deriv1 * self.kernel.deriveKernel(params, i)
		
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)		
		return raiseK(self.kernel.getTrainTestKernel(params, Xtest), self.polyDegree)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		return raiseK(self.kernel.getTestKernelDiag(params, Xtest), self.polyDegree)
		

class Poly2KernelHomo(PolyKernelHomo):
	def __init__(self, kernel):
		PolyKernelHomo.__init__(self, kernel)
		self.polyDegree = 2

class Poly3KernelHomo(PolyKernelHomo):
	def __init__(self, kernel):
		PolyKernelHomo.__init__(self, kernel)
		self.polyDegree = 3

		
		
		
		

		
		
		
		
class RBFKernel(Kernel):
	def __init__(self, X):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		if (X.shape[1] >= X.shape[0] or True): self.K_sq = sq_dist(self.X_scaled.T)
		else: self.K_sq = None

	def getNumParams(self): return 1
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']
		
		ell = np.exp(params[0])
		if (self.K_sq is None): K = sq_dist(self.X_scaled.T / ell)	#precompute squared distances
		else: K = self.K_sq / ell**2		
		self.cache['K_sq_scaled'] = K

		# # # #manual computation (just for sanity checks)
		# # # K1 = np.exp(-K / 2.0)
		# # # K2 = np.zeros((self.X_scaled.shape[0], self.X_scaled.shape[0]))
		# # # for i1 in xrange(self.X_scaled.shape[0]):
			# # # for i2 in xrange(i1, self.X_scaled.shape[0]):
				# # # diff = self.X_scaled[i1,:] - self.X_scaled[i2,:]
				# # # K2[i1, i2] = np.exp(-np.sum(diff**2) / (2*ell))
				# # # K2[i2, i1] = K2[i1, i2]				
		# # # print np.max((K1-K2)**2)
		# # # sys.exit(0)
		
		K_exp = np.exp(-K / 2.0)
		self.cache['getTrainKernel'] = K_exp
		self.saveParams(params)
		return K_exp
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)		
		return self.getTrainKernel(params) * self.cache['K_sq_scaled']
		
		#ell = np.exp(params[0])
		#if (self.K_sq is None): K = sq_dist(self.X_scaled.T / ell)	#precompute squared distances
		#else: K = self.K_sq / ell**2
		#return np.exp(-K / 2.0)*K
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		ell = np.exp(params[0])
		K = sq_dist(self.X_scaled.T/ell, (Xtest/np.sqrt(Xtest.shape[1])).T/ell)	#precompute squared distances
		return np.exp(-K / 2.0)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)		
		return np.ones(Xtest.shape[0])
		
		
		
		
		
		
class GaborKernel(Kernel):
	def __init__(self, X):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		if (X.shape[1] >= X.shape[0] or True): self.K_sq = sq_dist(self.X_scaled.T)
		else: self.K_sq = None
		
		#compute dp
		self.dp = np.zeros((X.shape[0], X.shape[0]))
		for d in xrange(self.X_scaled.shape[1]):
			self.dp += (np.outer(self.X_scaled[:,d], np.ones((1, self.X_scaled.shape[0]))) - np.outer(np.ones((self.X_scaled.shape[0], 1)), self.X_scaled[:,d]))

	def getNumParams(self): return 2
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		ell = np.exp(params[0])
		p = np.exp(params[1])
		
		#compute d2
		if (self.K_sq is None): d2 = sq_dist(self.X_scaled.T / ell)	#precompute squared distances
		else: d2 = self.K_sq / ell**2
		
		#compute dp
		dp = self.dp/p
		
		K = np.exp(-d2 / 2.0)
		return np.cos(2*np.pi*dp)*K
		
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		ell = np.exp(params[0])
		p = np.exp(params[1])
		
		#compute d2
		if (self.K_sq is None): d2 = sq_dist(self.X_scaled.T / ell)	#precompute squared distances
		else: d2 = self.K_sq / ell**2
		
		#compute dp
		dp = self.dp/p
		
		K = np.exp(-d2 / 2.0)
		if (i==0): return d2*K*np.cos(2*np.pi*dp)
		elif (i==1): return 2*np.pi*dp*np.sin(2*np.pi*dp)*K
		else: raise Exception('invalid parameter index:' + str(i))
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		ell = np.exp(params[0])
		p = np.exp(params[1])
		
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		d2 = sq_dist(self.X_scaled.T/ell, Xtest_scaled.T/ell)	#precompute squared distances
		
		#compute dp
		dp = np.zeros(d2.shape)
		for d in xrange(self.X_scaled.shape[1]):
			dp += (np.outer(self.X_scaled[:,d], np.ones((1, Xtest_scaled.shape[0]))) - np.outer(np.ones((self.X_scaled.shape[0], 1)), Xtest_scaled[:,d]))
		dp /= p
				
		K = np.exp(-d2 / 2.0)
		return np.cos(2*np.pi*dp)*K
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		return np.ones(Xtest.shape[0])
		
		
		
		
class RQKernel(Kernel):
	def __init__(self, X):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		if (X.shape[1] >= X.shape[0] or True): self.K_sq = sq_dist(self.X_scaled.T)
		else: self.K_sq = None

	def getNumParams(self): return 2
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		ell = np.exp(params[0])
		alpha = np.exp(params[1])
		
		if (self.K_sq is None): D2 = sq_dist(self.X_scaled.T / ell)	#precompute squared distances
		else: D2 = self.K_sq / ell**2
		return (1+0.5*D2/alpha)**(-alpha)
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		ell = np.exp(params[0])
		alpha = np.exp(params[1])
		if (self.K_sq is None): D2 = sq_dist(self.X_scaled.T / ell)	#precompute squared distances
		else: D2 = self.K_sq / ell**2
		if (i==0): return (1+0.5*D2/alpha)**(-alpha-1)*D2
		elif (i==1):
			K = (1+0.5*D2/alpha)
			return K**(-alpha) * (0.5*D2/K - alpha*(np.log(K)))
		else: raise Exception('invalid parameter index')
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		ell = np.exp(params[0])
		alpha = np.exp(params[1])
		D2 = sq_dist(self.X_scaled.T/ell, (Xtest/np.sqrt(Xtest.shape[1])).T/ell)	#precompute squared distances
		return (1+0.5*D2/alpha)**(-alpha)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		return np.ones(Xtest.shape[0])
		
		
		
		
		
		
		
		
class RBFRegionsKernel(Kernel):
	def __init__(self, X, regions):
		Kernel.__init__(self)		
		self.X_scaled = X.copy()
		for r in regions: self.X_scaled[:, r] /= np.sqrt(np.sum(r))
		self.regions = regions		
		
		self.K_sq = []
		for r in regions: self.K_sq.append(sq_dist(self.X_scaled[:,r].T))

	def getNumParams(self): return len(self.regions)
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']
		if ('K_sq_scaled' not in self.cache.keys()): self.cache['K_sq_scaled'] = [None for i in xrange(self.getNumParams())]
			
		ell = np.exp(params)
		K = 0
		for i in xrange(self.getNumParams()):
			if (self.sameParams(params, i)): K += self.cache['K_sq_scaled'][i]
			else:
				self.cache['K_sq_scaled'][i] = self.K_sq[i] / ell[i]**2
				K += self.cache['K_sq_scaled'][i]
		K_exp = np.exp(-K / 2.0)
		self.cache['getTrainKernel'] = K_exp
		self.saveParams(params)
		return K_exp
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		
		K_exp = self.getTrainKernel(params)
		#ell = np.exp(params)
		#return K_exp*(self.K_sq[i] / ell[i]**2)
		
		return K_exp * self.cache['K_sq_scaled'][i]
		
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		ell = np.exp(params)
		K = 0
		for r_i, r in enumerate(self.regions):
			Xtest_r = Xtest[:,r]/np.sqrt(np.sum(r))			
			K += sq_dist(self.X_scaled[:,r].T, Xtest_r.T) / ell[r_i]**2
		return np.exp(-K / 2.0)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)		
		return np.ones(Xtest.shape[0])

		
		
		
		
class AdditiveKernel(Kernel):
	def __init__(self, kernels, n):
		Kernel.__init__(self)
		self.kernels = kernels
		self.n = n
		self.prevKdimParams = None
		self.prevEEParams = None
		self.prevHyp0Params = None

	def getNumParams(self): return len(self.kernels) + np.sum([k.getNumParams() for k in self.kernels])
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']
		
		params_kernels = params[len(self.kernels):]
		EE = self.getEE(params_kernels)
		K=0				
		for i in xrange(len(self.kernels)):
			K += self.getScaledE(params, i, EE)
		self.cache['getTrainKernel'] = K
		self.saveParams(params)		
		return K
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		params_kernels = params[len(self.kernels):]
		
		#sf2 derivatives
		if (i < len(self.kernels)):
			EE = self.getEE(params_kernels)			
			#if (i==2): Z = 2*np.exp(2*params[i]) * EE[:,:,i+1]; print i, Z[:5, :5]; sys.exit(0)
			return 2*np.exp(2*params[i]) * EE[:,:,i+1]
		
		#params_kernel derivatives
		else:
			params_ind = len(self.kernels)
			for k_i, k in enumerate(self.kernels):
				numHyp = k.getNumParams()
				if (i not in xrange(params_ind, params_ind+numHyp)):
					params_ind += numHyp
					continue			
				
				#now we found our kernel
				dKj = k.deriveKernel(params[params_ind:params_ind+numHyp], i-params_ind)				
				Kd = self.Kdim(params_kernels)
				range1 = np.array(xrange(0,k_i), dtype=np.int)
				range2 = np.array(xrange(k_i+1, len(self.kernels)), dtype=np.int)
				Kd_nocov = Kd[:, :, np.concatenate((range1, range2))]
				E = elsympol(Kd_nocov, len(self.kernels)-1)  #R-1th elementary sym polyn				
				K=0
				for ii in xrange(len(self.kernels)):
					K += E[:,:,ii]*np.exp(2*params[ii])				
				#if (i==5): Z = dKj * K; print i, Z[:5, :5]; sys.exit(0)
				return dKj * K
		
		raise Exception('Invalid parameter')

		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)		
		params_kernels = params[len(self.kernels):]
		
		#compute Kd and EE
		Kd = np.zeros((self.n, Xtest[0].shape[0], len(self.kernels)))
		params_ind = 0
		kernel_paramsArr = params[len(self.kernels):]
		for k_i, k in enumerate(self.kernels):
			numHyp = k.getNumParams()
			kernelParams_range = np.array(xrange(params_ind, params_ind+numHyp), dtype=np.int)
			kernel_params = kernel_paramsArr[kernelParams_range]			
			Kd[:,:,k_i] = k.getTrainTestKernel(kernel_params, Xtest[k_i])
			params_ind += numHyp
		EE = elsympol(Kd, len(self.kernels))
		
		#compute K
		K=0				
		for i in xrange(len(self.kernels)): K += np.exp(2*params[i]) * EE[:,:,i+1]			
		
		return K		
		
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)		
		params_kernels = params[len(self.kernels):]
		
		#compute Kd and EE
		Kd = np.zeros((Xtest[0].shape[0], 1, len(self.kernels)))
		params_ind = 0
		kernel_paramsArr = params[len(self.kernels):]
		for k_i, k in enumerate(self.kernels):
			numHyp = k.getNumParams()
			kernelParams_range = np.array(xrange(params_ind, params_ind+numHyp), dtype=np.int)
			kernel_params = kernel_paramsArr[kernelParams_range]			
			Kd[:,0,k_i] = k.getTestKernelDiag(kernel_params, Xtest[k_i])
			params_ind += numHyp
		EE = elsympol(Kd, len(self.kernels))
		
		#compute K
		K=0				
		for i in xrange(len(self.kernels)): K += np.exp(2*params[i]) * EE[:,:,i+1]			
		return K
		
		
	def getEE(self, EEParams):
		if (self.prevEEParams is not None):
			if (EEParams.shape[0] == 0 or np.max(np.abs(EEParams-self.prevEEParams < self.epsilon))): return self.cache['EE']
		Kd = self.Kdim(EEParams)
		EE = elsympol(Kd, len(self.kernels))		
		self.prevEEParams = EEParams.copy()
		self.cache['EE'] = EE
		return EE
		

	def getScaledE(self, params, i, E):		
		if (self.prevHyp0Params is not None and np.abs(self.prevHyp0Params[i]-params[i]) < self.epsilon): return self.cache['E_scaled'][i]		
		if ('E_scaled' not in self.cache.keys()): self.cache['E_scaled'] = [None for j in xrange(len(self.kernels))]
		
		for j in xrange(len(self.kernels)):
			if (self.prevHyp0Params is not None and np.abs(self.prevHyp0Params[j]-params[j]) < self.epsilon): continue		
			E_scaled = E[:,:,j+1]*np.exp(2*params[j])
			self.cache['E_scaled'][j] = E_scaled
			
		self.prevHyp0Params = params.copy()		
		return self.cache['E_scaled'][i]
		
	def Kdim(self, kdimParams):
		if (self.prevKdimParams is not None and np.max(np.abs(kdimParams-self.prevKdimParams)) < self.epsilon): return self.cache['Kdim']
			
		K = np.zeros((self.n, self.n, len(self.kernels)))
		params_ind = 0
		for k_i, k in enumerate(self.kernels):
			numHyp = k.getNumParams()
			kernelParams_range = np.array(xrange(params_ind, params_ind+numHyp), dtype=np.int)			
			kernel_params = kdimParams[kernelParams_range]			
			if ((numHyp == 0 and 'Kdim' in self.cache) or (numHyp>0 and self.prevKdimParams is not None and np.max(np.abs(kernel_params-self.prevKdimParams[kernelParams_range])) < self.epsilon)):
				K[:,:,k_i] = self.cache['Kdim'][:,:,k_i]
			else:
				K[:,:,k_i] = k.getTrainKernel(kernel_params)				
			params_ind += numHyp
		self.prevKdimParams = kdimParams.copy()
		self.cache['Kdim'] = K
		return K




class MaternKernel(Kernel):
	def __init__(self, X):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		if (X.shape[1] >= X.shape[0] or True): self.K_sq = sq_dist(self.X_scaled.T * np.sqrt(self.d))
		else: self.K_sq = None
		self.m = lambda t,f: f(t)*np.exp(-t)
		self.dm = lambda t,df: df(t)*np.exp(-t)*t
		

	def getNumParams(self): return 1
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']
		
		ell = np.exp(params[0])
		if (self.K_sq is None): K = sq_dist(self.X_scaled.T * np.sqrt(self.d)/ell)	#precompute squared distances
		else: K = self.K_sq / ell**2
		self.cache['K_sq_scaled'] = K
		
		K_exp = self.m(np.sqrt(K), self.f)
		self.cache['getTrainKernel'] = K_exp
		self.saveParams(params)
		return K_exp
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		self.getTrainKernel(params)	#make sure that cache is updated
		return self.dm(np.sqrt(self.cache['K_sq_scaled']), self.df)
		
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		ell = np.exp(params[0])
		K = sq_dist(self.X_scaled.T*np.sqrt(self.d)/ell, ((Xtest/np.sqrt(Xtest.shape[1])).T)*np.sqrt(self.d)/ell)	#precompute squared distances
		return self.m(np.sqrt(K), self.f)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)		
		return self.m(np.zeros(Xtest.shape[0]), self.f)
		
class Matern1Kernel(MaternKernel):
	def __init__(self, X):
		self.d = 1
		self.f = lambda t: 1
		self.df = lambda t: 1
		MaternKernel.__init__(self, X)		
		
class Matern3Kernel(MaternKernel):
	def __init__(self, X):
		self.d = 3
		self.f = lambda t: 1+t
		self.df = lambda t: t
		MaternKernel.__init__(self, X)		
		
class Matern5Kernel(MaternKernel):
	def __init__(self, X):
		self.d = 5
		self.f = lambda t: 1+t*(1+t/3.0)
		self.df = lambda t: t*(1+t)/3.0
		MaternKernel.__init__(self, X)		
		

		
class PPKernel(Kernel):
	def __init__(self, X):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		if (X.shape[1] >= X.shape[0] or True): self.K_sq = sq_dist(self.X_scaled.T)
		else: self.K_sq = None
		self.j = np.floor(X.shape[1]/2.0)+self.v+1
		self.pp = lambda r,j,v,f:  np.maximum(1-r, 0)**(j+v) * self.f(r,j)
		self.dpp = lambda r,j,v,f: np.maximum(1-r, 0)**(j+v-1) * r * ((j+v)*f(r,j) - np.maximum(1-r,0)*self.df(r,j))

	def getNumParams(self): return 1
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']
		
		ell = np.exp(params[0])		
		if (self.K_sq is None): K = sq_dist(self.X_scaled.T / ell)	#precompute squared distances
		else: K = self.K_sq / ell**2		
		self.cache['K_sq_scaled'] = K
		
		K_exp = self.pp(K, self.j, self.v, self.f)
		self.cache['getTrainKernel'] = K_exp
		self.saveParams(params)
		return K_exp
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		self.getTrainKernel(params)	#make sure that cache is updates
		return self.dpp(self.cache['K_sq_scaled'], self.j, self.v, self.f)

	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		ell = np.exp(params[0])
		K = sq_dist(self.X_scaled.T/ell, (Xtest/np.sqrt(Xtest.shape[1])).T/ell)	#precompute squared distances
		return self.pp(K, self.j, self.v, self.f)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		K = np.zeros(Xtest.shape[0])
		return self.pp(K, self.j, self.v, self.f)
		
class PP0Kernel(PPKernel):
	def __init__(self, X):
		self.v = 0
		self.f = lambda r,j: 1
		self.df = lambda r,j: 0
		PPKernel.__init__(self, X)	
	
class PP1Kernel(PPKernel):
	def __init__(self, X):
		self.v = 1
		self.f = lambda r,j: 1+(j+1)*r
		self.df = lambda r,j: j+1
		PPKernel.__init__(self, X)	

class PP2Kernel(PPKernel):
	def __init__(self, X):
		self.v = 2
		self.f = lambda r,j: 1 + (j+2)*r + (j**2 + 4*j+ 3)/ 3.0*r**2
		self.df = lambda r,j: (j+2)   + 2*(  j**2+ 4*j+ 3)/ 3.0*r
		PPKernel.__init__(self, X)	

class PP3Kernel(PPKernel):
	def __init__(self, X):
		self.v = 3
		self.f = lambda r,j: 1 + (j+3)*r +  (6*j**2+36*j+45)/15.0*r**2 + (j**3+9*j**2+23*j+15)/15.0*r**3
		self.df = lambda r,j: (j+3)   + 2*(6*j**2+36*j+45)/15.0*r  + (j**3+9*j**2+23*j+15)/ 5.0*r**2
		PPKernel.__init__(self, X)	
		
		
		
		
		
class RealNNKernel(Kernel):
	def __init__(self, X, numUnits, dropoutProb=0):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		self.numUnits = numUnits
		self.numParams  = self.numUnits*X.shape[1]
		
		self.dropoutProb = dropoutProb
		self.dropoutMask = np.ones((X.shape[1], numUnits))
	
	def getNumParams(self): return self.numParams
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']
		
		if (self.dropoutProb > 0): self.dropoutMask = (np.random.random(size=self.dropoutMask.shape) > self.dropoutProb)

		nnX = self.applyNN(self.X_scaled, params) / np.sqrt(self.numUnits)
		K = nnX.dot(nnX.T)
		self.cache['getTrainKernel'] = K
		self.saveParams(params)
		return K
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		
		#find the relevant W
		numSNPs = self.X_scaled.shape[1]
		unitNum = i / numSNPs
		weightNum = i % numSNPs
		
		nnX_unitNum = self.applyNN(self.X_scaled, params, unitNum) / float(self.numUnits)
		w_deriv_relu = self.X_scaled[:, weightNum].copy()
		w_deriv_relu[nnX_unitNum <= 0] = 0
		
		K_deriv1 = np.outer(nnX_unitNum, w_deriv_relu)
		K_deriv = K_deriv1 + K_deriv1.T
		return K_deriv
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		self.dropoutMask = np.ones((self.X_scaled.shape[1], self.numUnits))					#no dropout
		nnX_train = self.applyNN(self.X_scaled, params)
		nnX_test  = self.applyNN(Xtest/np.sqrt(Xtest.shape[1]), params)
		return nnX_train.dot(nnX_test.T) / float((self.numUnits))
		
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		self.dropoutMask = np.ones((self.X_scaled.shape[1], self.numUnits))					#no dropout
		nnX = self.applyNN(Xtest/np.sqrt(Xtest.shape[1]), params) / np.sqrt(self.numUnits)
		return np.sum(nnX**2, axis=1)
		
	#compute the hidden layer output
	def applyNN(self, X, params, unitNum=None):
		W = np.reshape(params, (X.shape[1], self.numUnits), order='F') * self.dropoutMask
		if (unitNum is not None): W  = W[:, unitNum]
		XW = X.dot(W)
		relu = np.maximum(XW, 0)
		return relu
		
		
		
#LD kernel with exponential decay and a bias term
class LDKernel(Kernel):
	def __init__(self, X, pos):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		d = pos.shape[0]
		self.D = np.abs(np.tile(np.column_stack(pos).T, (1, d)) - np.tile(pos, (d, 1))) / 100000.0
		
	def getNumParams(self): return 2
		
	def getTrainKernel(self, params):
		self.checkParams(params)
		b = np.exp(params[0])
		k = np.exp(params[1])
		M = 1.0 / (b + np.exp(k*self.D))
		return self.X_scaled.dot(M).dot(self.X_scaled.T)
		
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		b = np.exp(params[0])
		k = np.exp(params[1])
		
		expD = np.exp(k*self.D)
		d = -(b+expD)**(-2)
		if (i==0): Md = d*b
		else: Md = d*self.D*expD * k	#*k is because of the exp
		return self.X_scaled.dot(Md).dot(self.X_scaled.T)
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		b = np.exp(params[0])
		k = np.exp(params[1])
		M = 1.0 / (b + np.exp(k*self.D))
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return self.X_scaled.dot(M).dot(Xtest_scaled.T)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		b = np.exp(params[0])
		k = np.exp(params[1])
		M = 1.0 / (b + np.exp(k*self.D))
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return np.diag((Xtest_scaled).dot(M).dot(Xtest_scaled.T))
		
		
		
		
#LD kernel with exponential decay but no bias term
class LDKernel2(Kernel):
	def __init__(self, X, pos):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		d = pos.shape[0]
		self.D = np.abs(np.tile(np.column_stack(pos).T, (1, d)) - np.tile(pos, (d, 1))) / 100000.0
		
	def getNumParams(self): return 1
		
	def getTrainKernel(self, params):
		self.checkParams(params)		
		k = np.exp(params[0])
		M = 1.0 / np.exp(k*self.D)
		return self.X_scaled.dot(M).dot(self.X_scaled.T)
		
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		k = np.exp(params[0])
		
		expD = np.exp(k*self.D)
		d = -expD**(-2)		
		Md = d*self.D*expD * k	#*k is because of the exp
		return self.X_scaled.dot(Md).dot(self.X_scaled.T)
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		k = np.exp(params[0])
		M = 1.0 / np.exp(k*self.D)
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return self.X_scaled.dot(M).dot(Xtest_scaled.T)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)		
		k = np.exp(params[0])
		M = 1.0 / np.exp(k*self.D)
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return np.diag((Xtest_scaled).dot(M).dot(Xtest_scaled.T))
		
		
#LD kernel with linear decay
class LDKernel3(Kernel):
	def __init__(self, X, pos):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		d = pos.shape[0]
		self.D = np.abs(np.tile(np.column_stack(pos).T, (1, d)) - np.tile(pos, (d, 1))) / 100000.0
		
	def getNumParams(self): return 1
		
	def getTrainKernel(self, params):
		self.checkParams(params)		
		k = np.exp(params[0])
		M = 1.0 / (1.0 + k*self.D)
		return self.X_scaled.dot(M).dot(self.X_scaled.T)
		
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		k = np.exp(params[0])
		
		Md = -(1+k*self.D)**(-2) * self.D * k		
		return self.X_scaled.dot(Md).dot(self.X_scaled.T)
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		k = np.exp(params[0])
		M = 1.0 / (1.0 + k*self.D)
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return self.X_scaled.dot(M).dot(Xtest_scaled.T)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)		
		k = np.exp(params[0])
		M = 1.0 / (1.0 + k*self.D)
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return np.diag((Xtest_scaled).dot(M).dot(Xtest_scaled.T))
		
		
		
		
		
#LD kernel with polynomial decay
class LDKernel4(Kernel):
	def __init__(self, X, pos):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		d = pos.shape[0]
		self.D = np.abs(np.tile(np.column_stack(pos).T, (1, d)) - np.tile(pos, (d, 1))) / 100000.0
		
	def getNumParams(self): return 2
		
	def getTrainKernel(self, params):
		self.checkParams(params)
		p = np.exp(params[0])
		k = np.exp(params[1])		
		M = 1.0 / (1.0 + k*(self.D**p))
		return self.X_scaled.dot(M).dot(self.X_scaled.T)
		
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		p = np.exp(params[0])
		k = np.exp(params[1])

		d = -(1+k*self.D)**(-2)
		if (i==0): Md = d * k * p * self.D**(p-1) * p
		else: Md = d * self.D**p * k		
		return self.X_scaled.dot(Md).dot(self.X_scaled.T)
		
	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		p = np.exp(params[0])
		k = np.exp(params[1])
		M = 1.0 / (1.0 + k*(self.D**p))
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return self.X_scaled.dot(M).dot(Xtest_scaled.T)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)		
		p = np.exp(params[0])
		k = np.exp(params[1])
		M = 1.0 / (1.0 + k*(self.D**p))
		Xtest_scaled = Xtest/np.sqrt(Xtest.shape[1])
		return np.diag((Xtest_scaled).dot(M).dot(Xtest_scaled.T))

		
		
		
class NNKernel(Kernel):
	def __init__(self, X):
		Kernel.__init__(self)
		self.X_scaled = X/np.sqrt(X.shape[1])
		self.sx = 1 + np.sum(self.X_scaled**2, axis=1)
		self.S = 1 + self.X_scaled.dot(self.X_scaled.T)
	
	def getNumParams(self): return 1
	
	def getTrainKernel(self, params):
		self.checkParams(params)
		if (self.sameParams(params)): return self.cache['getTrainKernel']				
		ell2 = np.exp(2*params[0])
		
		sqrt_ell2PSx = np.sqrt(ell2+self.sx)
		K = self.S / np.outer(sqrt_ell2PSx, sqrt_ell2PSx)
		self.cache['K'] = K
		K_arcsin = np.arcsin(K)
		
		self.cache['getTrainKernel'] = K_arcsin
		self.saveParams(params)
		return K_arcsin
		
	def deriveKernel(self, params, i):
		self.checkParamsI(params, i)
		self.getTrainKernel(params)	#make sure that cache is updated
		ell2 = np.exp(2*params[0])
		
		vx = self.sx / (ell2+self.sx)
		vxDiv2 = vx/2.0
		n = self.X_scaled.shape[0]
		V = np.tile(np.column_stack(vxDiv2).T, (1, n)) + np.tile(vxDiv2.T, (n, 1))
		
		K = self.cache['K']
		return -2*(K-K*V) / np.sqrt(1-K**2)
		

	def getTrainTestKernel(self, params, Xtest):
		self.checkParams(params)
		ell2 = np.exp(2*params[0])
		
		z = Xtest / np.sqrt(Xtest.shape[1])
		S = 1 + self.X_scaled.dot(z.T)
		sz = 1 + np.sum(z**2, axis=1)
		sqrtEll2Psx = np.sqrt(ell2+self.sx)
		sqrtEll2Psz = np.sqrt(ell2+sz)
		K = S / np.outer(sqrtEll2Psx, sqrtEll2Psz)
		return np.arcsin(K)
		
	def getTestKernelDiag(self, params, Xtest):
		self.checkParams(params)
		ell2 = np.exp(2*params[0])
		
		z = Xtest / np.sqrt(Xtest.shape[1])
		sz = 1 + np.sum(z**2, axis=1)
		K = sz / (sz + ell2)
		return np.arcsin(K)

