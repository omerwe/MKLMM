import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import time
import sys
import sklearn.linear_model
import scipy.optimize as optimize
import gpUtils
import kernels
np.set_printoptions(precision=4, linewidth=200)

class MKLMM:

	def __init__(self, verbose=False):
		self.verbose = verbose
		pass

		
	def fit(self, X, C, y, regions, kernelType, reml=True, maxiter=100):
	
		#construct a list of kernel names (one for each region) 
		if (kernelType == 'adapt'): kernelNames = self.buildKernelAdapt(X, C, y, regions, reml, maxiter)
		else: kernelNames = [kernelType] * len(regions)			
		
		#perform optimization
		kernelObj, hyp_kernels, sig2e, fixedEffects = self.optimize(X, C, y, kernelNames, regions, reml, maxiter)
		
		#compute posterior distribution
		Ktraintrain = kernelObj.getTrainKernel(hyp_kernels)
		post = self.infExact_scipy_post(Ktraintrain, C, y, sig2e, fixedEffects)
		
		#fix intercept if phenotype is binary
		if (len(np.unique(y)) == 2):			
			controls = (y<y.mean())
			cases = ~controls
			meanVec = C.dot(fixedEffects)
			mu, var = self.getPosteriorMeanAndVar(np.diag(Ktraintrain), Ktraintrain, post, meanVec)										
			fixedEffects[0] -= optimize.minimize_scalar(self.getNegLL, args=(mu, np.sqrt(sig2e+var), controls, cases), method='brent').x				
		
		#construct trainObj
		trainObj = dict([])
		trainObj['sig2e'] = sig2e
		trainObj['hyp_kernels'] = hyp_kernels
		trainObj['fixedEffects'] = fixedEffects		
		trainObj['kernelNames'] = kernelNames
		
		return trainObj

		
		
	def predict(self, X_train, C_train, y_train, regions, X_test, testC, trainObj):
		
		if (len(regions) != len(trainObj['kernelNames'])): raise Exception('#regions doesn''t match the training data')
		kernelObj, _ = self.buildKernel(X_train, trainObj['kernelNames'], regions, 1.0)
		K = kernelObj.getTrainKernel(trainObj['hyp_kernels'])
		post = self.infExact_scipy_post(K, C_train, y_train, trainObj['sig2e'],  trainObj['fixedEffects'])
		mu, var = self.predictMuAndVar(X_test, trainObj['kernelNames'], regions, kernelObj, post, trainObj['hyp_kernels'], testC, trainObj['fixedEffects'])
		return mu, var + trainObj['sig2e']
		
		
	######################################## auxilary methods ################################################
	
	def getNegLL(self, t, mu, sqrtVar, controls, cases):
		z = (mu-t) / sqrtVar
		logProbControls = stats.norm(0,1).logcdf(-z)
		logProbCases    = stats.norm(0,1).logcdf(z)				
		ll = logProbControls[controls].sum() + logProbCases[cases].sum()
		return -ll
		

	def predictMuAndVar(self, X, kernelNames, regions, kernelObj, post, hyp_kernels, testC, fixedEffects):
		X_test = []
		for i in xrange(len(regions)):
			r = regions[i]
			if (kernelNames[i][-4:] == '_lin'): X_test += [X[:, r], X[:, r]]
			else: X_test.append(X[:, r])
		
		Ktraintest = kernelObj.getTrainTestKernel(hyp_kernels, X_test)
		diagKtesttest = kernelObj.getTestKernelDiag(hyp_kernels, X_test)		
		
		meanVec = testC.dot(fixedEffects)
		mu, var = self.getPosteriorMeanAndVar(diagKtesttest, Ktraintest, post, meanVec)
		return mu, var
		
		
	def buildKernel(self, X, kernelNames, regions, yVar):
		numVarComp = len(regions)
		hyp0_kernels = []		

		kernelsList = []
		for r_i, r in enumerate(regions):
			regionSize = r.sum()
			kernelName = kernelNames[r_i]

			#choose kernel
			if (kernelName == 'lin'):
				kernel = kernels.linearKernel(X[:, r])
			elif (kernelName == 'rbf_lin'):
				kernel1 = kernels.ScaledKernel(kernels.RBFKernel(X[:, r]))
				hyp0_kernels.append(np.log(1.0))	#ell
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel1)						
				kernel2 = kernels.ScaledKernel(kernels.linearKernel(X[:, r]))
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel2)
				continue
			elif (kernelName == 'poly2_lin'):
				kernel1 = kernels.ScaledKernel(kernels.Poly2KernelHomo(kernels.linearKernel(X[:, r])))						
				hyp0_kernels.append(0.5*np.log(0.5 / numVarComp))	#scaling hyp
				kernelsList.append(kernel1)						
				kernel2 = kernels.ScaledKernel(kernels.linearKernel(X[:, r]))
				hyp0_kernels.append(0.5*np.log(0.5 / numVarComp))	#scaling hyp
				kernelsList.append(kernel2)
				continue
			elif (kernelName == 'poly3_lin'):
				kernel1 = kernels.ScaledKernel(kernels.Poly3KernelHomo(kernels.linearKernel(X[:, r])))
				hyp0_kernels.append(0.5*np.log(0.5 / numVarComp))	#scaling hyp
				kernelsList.append(kernel1)
				kernel2 = kernels.ScaledKernel(kernels.linearKernel(X[:, r]))
				hyp0_kernels.append(0.5*np.log(0.5 / numVarComp))	#scaling hyp
				kernelsList.append(kernel2)
				continue
			elif (kernelName in ['nn_lin']):
				kernel1 = kernels.ScaledKernel(kernels.NNKernel(X[:, r]))
				hyp0_kernels.append(np.log(1.0))	#ell
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel1)						
				kernel2 = kernels.ScaledKernel(kernels.linearKernel(X[:, r]))
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel2)
				continue
			elif (kernelName == 'matern5_lin'):
				kernel1 = kernels.ScaledKernel(kernels.Matern5Kernel(X[:, r]))
				hyp0_kernels.append(np.log(1.0))	#ell
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel1)						
				kernel2 = kernels.ScaledKernel(kernels.linearKernel(X[:, r]))
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel2)
				continue
			elif (kernelName == 'matern3_lin'):
				kernel1 = kernels.ScaledKernel(kernels.Matern3Kernel(X[:, r]))
				hyp0_kernels.append(np.log(1.0))	#ell
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel1)
				kernel2 = kernels.ScaledKernel(kernels.linearKernel(X[:, r]))
				hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
				kernelsList.append(kernel2)
				continue
			elif (kernelName == 'poly2'):
				kernel = kernels.linearKernel(X[:, r])
				kernel = kernels.Poly2Kernel(kernel)
				hyp0_kernels.append(np.log(1.0))	#bias hyp
			elif (kernelName == 'poly3'):
				kernel = kernels.linearKernel(X[:, r])
				kernel = kernels.Poly3Kernel(kernel)
				hyp0_kernels.append(np.log(1.0))	#bias hyp
			elif (kernelName == 'rbf'):
				kernel = kernels.RBFKernel(X[:, r])
				hyp0_kernels.append(np.log(1.0))	#ell
			elif (kernelName == 'gabor'):
				kernel = kernels.GaborKernel(X[:, r])
				hyp0_kernels += [np.log(1.0), np.log(1.0)]	#ell and p
			elif (kernelName == 'nn'):
				kernel = kernels.NNKernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			elif (kernelName == 'rq'):
				kernel = kernels.RQKernel(X[:, r])
				hyp0_kernels += [np.log(1.0), np.log(1.0)]	#ell and alpha
			elif (kernelName == 'matern1'):
				kernel = kernels.Matern1Kernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			elif (kernelName == 'matern3'):
				kernel = kernels.Matern3Kernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			elif (kernelName == 'matern5'):
				kernel = kernels.Matern5Kernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			elif (kernelName == 'pp0'):
				kernel = kernels.PP0Kernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			elif (kernelName == 'pp1'):
				kernel = kernels.PP1Kernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			elif (kernelName == 'pp2'):
				kernel = kernels.PP2Kernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			elif (kernelName == 'pp3'):
				kernel = kernels.PP3Kernel(X[:, r])
				hyp0_kernels += [np.log(1.0)]	#ell
			else: raise Exception('unknown kernel: ' + kernelName)
			
			#scale kernel				
			kernel = kernels.ScaledKernel(kernel)
			hyp0_kernels.append(0.5*np.log(0.5*yVar / numVarComp))	#scaling hyp
			kernelsList.append(kernel)
			
		if (kernelName in ['add']):
			combinedKernel = kernels.AdditiveKernel(kernelsList, y.shape[0])
			hyp0_kernels = np.concatenate((np.zeros(len(kernelsList)), hyp0_kernels))
		else: combinedKernel = kernels.SumKernel(kernelsList)

		return combinedKernel, hyp0_kernels
		
		
		
		
	def buildKernelAdapt(self, X, C, y, regions, reml=True, maxiter=100):
	
		#prepare initial values for sig2e and for fixed effects
		hyp0_sig2e, hyp0_fixedEffects = self.getInitialHyps(X, C, y)	
		
		
		bestKernelNames = []
		kernelsListAll 	= []
		hyp_kernels = []

		funcToSolve = self.infExact_scipy
		yVar = y.var()		
		
		for r_i, r in enumerate(regions):
		
			#if (r_i == 0): kernelsToTry = ['lin']
			#else:
			#	kernelsToTry = ['lin', 'poly2_lin', 'rbf_lin', 'nn_lin']		
			kernelsToTry = ['lin', 'poly2_lin', 'rbf_lin', 'nn_lin']
			if self.verbose:
				print
				print 'selecting a kernel for region', r_i, 'with', r.sum(), 'SNPs'
		
			#add linear kernel
			X_lastRegion = X[:, r]
			linKernel = kernels.linearKernel(X_lastRegion)
			kernelsListAll.append(kernels.ScaledKernel(linKernel))
			kernelsListAll.append(None)
			
			bestFun = np.inf
			bestKernelName = None
			best_hyp0 = None
			bestKernel = None
			bestPval = np.inf
		
			#iterate over every possible kernel			
			for kernelToTry in kernelsToTry:
				hyp0 = [0.5*np.log(0.5*yVar)]
				if self.verbose: print 'Testing kernel:', kernelToTry
				
				#create the kernel
				if (kernelToTry == 'lin'):
					kernel = None
					df = None
				elif (kernelToTry == 'rbf_lin'):
					kernel = kernels.RBFKernel(X_lastRegion)
					hyp0.append(np.log(1.0))	#ell
					df = 2
				elif (kernelToTry == 'nn_lin'):
					kernel = kernels.NNKernel(X_lastRegion)
					hyp0.append(np.log(1.0))	#ell
					df = 2
				elif (kernelToTry == 'poly2_lin'):
					kernel = kernels.Poly2KernelHomo(linKernel)				
					df = 1
				else:
					raise Exception('unrecognized kernel name')
					
				if (kernel is not None):
					#scale the kernel
					kernel = kernels.ScaledKernel(kernel)
					hyp0.append(0.5*np.log(0.5*yVar))	#scaling hyp
					
					#add the kernel as the final kernel in the kernels list
					kernelsListAll[-1] = kernel
					sumKernel = kernels.SumKernel(kernelsListAll)
				else:
					sumKernel = kernels.SumKernel(kernelsListAll[:-1])
				
				#test log likelihood obtained with this kernel for this region							
				args = (sumKernel, C, y, reml)
				self.optimization_counter=0
				hyp0_all = np.concatenate((hyp0_sig2e, hyp0_fixedEffects, hyp_kernels+hyp0))
				optObj = gpUtils.minimize(hyp0_all, funcToSolve, -maxiter, *args)
				if (not optObj.success):
					print 'Optimization status:', optObj.status
					print 'optimization message:', optObj.message
					raise Exception('optimization failed')
					
				print 'final LL: %0.5e'%(-optObj.fun)
				if (kernelToTry == 'lin'):
					linLL = -optObj.fun
					pVal = 1.0
				else:
					llDiff = -optObj.fun - linLL
					if (llDiff < 0): pVal = 1.0
					else: pVal = 0.5*stats.chi2(df).sf(llDiff)
					print 'llDiff: %0.5e'%llDiff, 'pVal:%0.5e'%pVal
					
				if (kernelToTry == 'lin' or (pVal < bestPval and (len(kernelsToTry)==1 or pVal < 0.05/(len(kernelsToTry)-1)))):
					bestOptObj = optObj
					bestPval = pVal
					bestKernelName = kernelToTry				
					best_hyp0 = hyp0
					best_sumKernel = sumKernel
					bestKernel = kernel		
			
			if (bestKernel is not None): kernelsListAll[-1] = bestKernel
			else: kernelsListAll = kernelsListAll[:-1]		
			hyp_kernels += best_hyp0
			bestKernelNames.append(bestKernelName)
			
			if self.verbose: print 'selected kernel:', bestKernelName
		
		if self.verbose:
			print 'selected kernels:', bestKernelNames
			print			
			
		return bestKernelNames
		
		
		
		
		
	def getInitialHyps(self, X, C, y):
		self.logdetXX  = np.linalg.slogdet(C.T.dot(C))[1]
		
		hyp0_sig2e = [0.5*np.log(0.5*y.var())]
		Linreg = sklearn.linear_model.LinearRegression(fit_intercept=False, normalize=False, copy_X=False)
		Linreg.fit(C, y)
		hyp0_fixedEffects = Linreg.coef_		
		return hyp0_sig2e, hyp0_fixedEffects
	

		
		
	def optimize(self, X, C, y, kernelNames, regions, reml=True, maxiter=100):
		methodName = ('REML' if reml else 'ML')
		if self.verbose: print 'Finding MKLMM',  methodName, 'parameters for', len(regions), 'regions with lengths:', [np.sum(r) for r in regions]
		
		#prepare initial values for sig2e and for fixed effects
		hyp0_sig2e, hyp0_fixedEffects = self.getInitialHyps(X, C, y)
		
		#build kernel and train a model
		t0 = time.time()				
		kernel, hyp0_kernels = self.buildKernel(X, kernelNames, regions, y.var())
			
		hyp0 = np.concatenate((hyp0_sig2e, hyp0_fixedEffects, hyp0_kernels))		
		args = (kernel, C, y, reml)
		funcToSolve = self.infExact_scipy

		# # #check gradient correctness
		# # if (len(hyp0) < 10):
			# # self.optimization_counter=0
			# # likFunc  = lambda hyp: funcToSolve(hyp, kernel, C, y, reml)[0]
			# # gradFunc = lambda hyp: funcToSolve(hyp, kernel, C, y, reml)[1]
			# # err = optimize.check_grad(likFunc, gradFunc, hyp0)
			# # print 'gradient error:', err
		
		if self.verbose: print 'Beginning Optimization'
		self.optimization_counter=0			
		optObj = gpUtils.minimize(hyp0, funcToSolve, -maxiter, *args)
			
		if (not optObj.success):
			print 'Optimization status:', optObj.status
			print 'optimization message:', optObj.message
			raise Exception('Optimization failed with message: ' + optObj.message)

		sig2e = np.exp(2*optObj.x[0])
		fixedEffects = optObj.x[1:C.shape[1]+1]
		hyp_kernels = optObj.x[C.shape[1]+1:]
		kernelObj = kernel

		if self.verbose:
			print 'done in %0.2f'%(time.time()-t0), 'seconds'
			print 'sig2e:', sig2e
			print 'Fixed effects:', fixedEffects
			if (hyp_kernels.shape[0] < 18): print 'kernel params:', hyp_kernels
			
		return kernelObj, hyp_kernels, sig2e, fixedEffects
		
		
		
		
	#convention: hyp[0] refers to sig2e, hyp[1:1+C.shape[1]+1] refer to fixed effects, hyp[self.trainCovars.shape[1]+1:]] refers to kernels
	def infExact_scipy(self, hyp, kernel, C, y, reml=True):
	
		n = y.shape[0]
		
		#mean vector
		fixedEffects = hyp[1:1+C.shape[1]]
		m = C.dot(fixedEffects)
		
		#build  kernel
		hyp_kernels = hyp[C.shape[1]+1:]
		K = kernel.getTrainKernel(hyp_kernels)
		
		sn2 = np.exp(2*hyp[0])		  	   #noise variance of likGauss			
		if sn2<1e-6:                       #very tiny sn2 can lead to numerical trouble
			L = la.cholesky(K + sn2*np.eye(n), overwrite_a=True, check_finite=False)    #Cholesky factor of covariance with noise
			sl =   1
		else:
			L = la.cholesky(K/sn2 + np.eye(n), overwrite_a=True, check_finite=False)	   #Cholesky factor of B
			sl = sn2               	   
		alpha = self.solveChol(L, y-m, overwrite_b=False) / sl

		#log likelihood
		nlZ = (y-m).dot(alpha/2.0) + np.sum(np.log(np.diag(L))) + n*np.log(2*np.pi*sl)/2.0   #-log marg lik		
		invKy = self.solveChol(L, np.eye(n))/sl
		if reml:
			d = C.shape[1]
			alpha2 = self.solveChol(L, C, overwrite_b=False) / sl
			XT_InvKy_X = C.T.dot(alpha2)			
			_, logDetXKindX = np.linalg.slogdet(XT_InvKy_X)			
			invXTInvKX = la.inv(XT_InvKy_X)
			X_invKy = C.T.dot(invKy)
			nlZ += 0.5*(logDetXKindX - self.logdetXX - d*np.log(2.0*np.pi))
			
		
			
			
		#derivatives		
		Q = invKy - np.outer(alpha, alpha)	#precompute for convenience	
		grad = np.zeros(hyp.shape[0])
		grad[0] = sn2*np.trace(Q)										#derivariate of sig2e
		if reml:
			gradMat = sn2*invXTInvKX.dot(X_invKy.dot(X_invKy.T))
			grad[0] -= np.trace(gradMat)
			
		#derivatives of fixed effects		
		grad[1:1+C.shape[1]] = -C.T.dot(alpha)
		
		#derivatives of variance components
		for i in xrange(hyp_kernels.shape[0]):
			halfDeriv = kernel.deriveKernel(hyp_kernels, i) / 2.0
			grad[i+C.shape[1]+1] = np.sum(Q*halfDeriv)
			if reml:
				gradMat = invXTInvKX.dot(X_invKy.dot(halfDeriv).dot(X_invKy.T))
				grad[i+C.shape[1]+1] -= np.trace(gradMat)
		
		if self.verbose:
			self.optimization_counter+=1		
			if (self.optimization_counter % 10 == 0):
				print 'Iteration', self.optimization_counter, '-LL:', nlZ

		return (nlZ, grad)

		
		
		
	def infExact_scipy_post(self, K, covars, y, sig2e, fixedEffects):
		n = y.shape[0]

		#mean vector
		m = covars.dot(fixedEffects)
		
		if (K.shape[1] < K.shape[0]): K_true = K.dot(K.T)
		else: K_true = K
		
		if sig2e<1e-6:
			L = la.cholesky(K_true + sig2e*np.eye(n), overwrite_a=True, check_finite=False)    	 #Cholesky factor of covariance with noise
			sl =   1
			pL = -self.solveChol(L, np.eye(n))         									 		 #L = -inv(K+inv(sW^2))
		else:
			L = la.cholesky(K_true/sig2e + np.eye(n), overwrite_a=True, check_finite=False)	  	 #Cholesky factor of B
			sl = sig2e               	   
			pL = L                		   												 		 #L = chol(eye(n)+sW*sW'.*K)
		alpha = self.solveChol(L, y-m, overwrite_b=False) / sl
			
		post = dict([])	
		post['alpha'] = alpha					  										  		#return the posterior parameters
		post['sW'] = np.ones(n) / np.sqrt(sig2e)									  			#sqrt of noise precision vector
		post['L'] = pL
		return post
		
		
	def solveChol(self, L, B, overwrite_b=True):
		cholSolve1 = la.solve_triangular(L, B, trans=1, check_finite=False, overwrite_b=overwrite_b)
		cholSolve2 = la.solve_triangular(L, cholSolve1, check_finite=False, overwrite_b=True)
		return cholSolve2
		
		
	def getPosteriorMeanAndVar(self, diagKTestTest, KtrainTest, post, intercept=0):
		L = post['L']
		if (np.size(L) == 0): raise Exception('L is an empty array') #possible to compute it here
		Lchol = np.all((np.all(np.tril(L, -1)==0, axis=0) & (np.diag(L)>0)) & np.isreal(np.diag(L)))
		ns = diagKTestTest.shape[0]
		nperbatch = 5000
		nact = 0
		
		#allocate mem
		fmu = np.zeros(ns)	#column vector (of length ns) of predictive latent means
		fs2 = np.zeros(ns)	#column vector (of length ns) of predictive latent variances
		while (nact<(ns-1)):
			id = np.arange(nact, np.minimum(nact+nperbatch, ns))
			kss = diagKTestTest[id]		
			Ks = KtrainTest[:, id]
			if (len(post['alpha'].shape) == 1):
				try: Fmu = intercept[id] + Ks.T.dot(post['alpha'])
				except: Fmu = intercept + Ks.T.dot(post['alpha'])
				fmu[id] = Fmu
			else:
				try: Fmu = intercept[id][:, np.newaxis] + Ks.T.dot(post['alpha'])
				except: Fmu = intercept + Ks.T.dot(post['alpha'])
				fmu[id] = Fmu.mean(axis=1)
			if Lchol:
				V = la.solve_triangular(L, Ks*np.tile(post['sW'], (id.shape[0], 1)).T, trans=1, check_finite=False, overwrite_b=True)
				fs2[id] = kss - np.sum(V**2, axis=0)                       #predictive variances						
			else:
				fs2[id] = kss + np.sum(Ks * (L.dot(Ks)), axis=0)		   #predictive variances
			fs2[id] = np.maximum(fs2[id],0)  #remove numerical noise i.e. negative variances		
			nact = id[-1]    #set counter to index of last processed data point
			
		return fmu, fs2
			
			
			
			

	
			
