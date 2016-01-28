# MKLMM
Multi Kernel Linear Mixed Models for Complex Phenotype Prediction

MKLMM is a Python package for predition of complex phenotypes from single nucleotide polymorphism (SNP) data that can model genetic interactions by using multi-kernel-learning techniques. The model is a generalization of [Adaptive MultiBLUP](http://genome.cshlp.org/content/early/2014/06/24/gr.169375.113.abstract), which divides the genome into several regions and infers a different variance component for every region. MKLMM improves upon MultiBLUP by additionally modeling genetic interactions, which are modeled by non-linear variance components (or kernels), where all model parameters are inferred jointly via restricted maximum likelihood.

MKLMM is particularly suitable for modeling complex local interactions between nearby variants. MKLMM-Adapt is a variant of MKLMM which automatically infers interaction patterns across multiple genomic regions. 

Several parts of the code are based on code translated from the [GPML toolbox](http://www.gaussianprocess.org/gpml/code/matlab/) and [Fast-LMM](https://github.com/MicrosoftGenomics/FaST-LMM).

------------------
Installation
------------------
MKLMM is designed to work in Python 2.7, and depends on the following freely available Python package:
* [Numpy](http://www.numpy.org/) and [Scipy](http://www.scipy.org/)
* [Scikits-learn](http://scikit-learn.org/stable/)
* [PySnpTools](https://github.com/MicrosoftGenomics/PySnpTools)

Typically, the packages can be installed with the command "pip install --user <package_name>".

MKLMM is particularly easy to use with the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda). The [numerically optimized version](http://docs.continuum.io/mkl-optimizations/index) of Anaconda can speed LEAP up by several orders of magnitude.
Alternatively (if numerically optimized Anaconda can't be installed), for very fast performance it is recommended to have an optimized version of Numpy/Scipy [installed on your system](http://www.scipy.org/scipylib/building), using optimized numerical libraries such as [OpenBLAS](http://www.openblas.net) or [Intel MKL](https://software.intel.com/en-us/intel-mkl) (see [Compilation instructions for scipy with Intel MKL)](https://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl).

Once all the prerequisite packages are installed, MKLMM can be installed on a git-enabled machine by typing:
```
git clone https://github.com/omerwe/MKLMM
```
```
unzip example.zip
```

The project can also be downloaded as a zip file from the Github website.

------------------
Usage Overview
------------------
MKLMM works with [binary Plink format](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed), and with phenotype and covariate files written in Plink format. 

There are two files that need to be executed:

1. rank_regions.py:  This file divides the genome into small regions (mean length 75Kb) and ranks them. It outputs a file that reports the first and last SNP in each region, and the region score.

2. mklmm_wrapper.py: This file trains a model and predicts phenotypes for a set of individuals. The file can be invoked in train mode, test mode or both. When invoked in train mode, it creates an output file with the learned REML parameters. When invoked in test mode, it creates an output file which lists the posterior mean and variance of the estimated phenotypes of all individuals.


The list of available options for both files can be seen by typing
```
python <file_name> --help
```

For an example, please run:
```
python mklmm_wrapper.py --bfile_train example/train --pheno example/train.phe --bfile example/test --regions example/regions.txt --numRegions 2 --kernel lin --out example/predictions.txt
```
This will train a model on the individuals in the file example/train.bed, will perform prediction on individuals in the file example/test.bed and will write the output to the file example/predictions.txt. The model will use two regions: One genome-wide region spanning all SNPs, and an additional region that obtained the highest log-likelihood in regions-ranking stage.

------------------
Detailed Instructions
------------------
####Ranking of regions
To rank regions, one should type:
```
python rank_regions.py --bfile_train <bfile> --pheno_train <phenotype file> --meanLen <mean region length> --out <output file> --covar_train <covariates file>
```
The bfile_train and phenotype files should contain only train set individuals, to prevent leakage. The covar_train flag is optional, and can be used to specify covariates wich will be modeled with fixed effects, in Plink format. Note that MKLMM automatically creates a covariate that is an intercept covariate, so there is no need to add a column of ones.

It is also possible to specify a bfile with test individuals (using the flag --bfile). This is useful for removal of top principal components from the genotypes, which can prevent confounding due to population structure (see details below). For an effective removal, it is required to compute the principal components using both the train and test individuals. This does not present any form of leakage, because the phenotypes of the test individuals are not known at any stage.
Ranking of regions is typically very fast, owing to the fast LMM inference algorithm of the FastLMM method.


The output file contains a row for every region, where each row has three entries. These entries correspond to the first and last SNP in the region (where the first SNP is numbered as 0), and to the log likeliood of the phenotype when using an LMM whose covariance matrix consists of only the SNPs in that region.

####Training an MKLMM model
To train a model, one should type:
```
python mklmm_wrapper.py --bfile_train <bfile> --pheno_train <phenotype file> --covar_train <covariates file> --regions <regions file> --numRegions <#regions to use> --train_out <output training file> --kernel <kernel type> 
```

The bfile_train, phenotype file and covar_train files should be the same as those provided in the first stage. The regions file is the output of the first stage. numRegions specifies the number of kernels that will get assigned their own kernel. For example, when numRegions=0, only a single genome-wide kernel is used, and the model thus becomes equivalent to GBLUP. The code automatically merges adjacent regions with a high log likelihood together. The parameter train_out specifies the output file which will contain the learned parameters of the kernels and the fixed effects.
The --kernel flag specifies the type of kernel that will be used, among several options:

* lin - a linear kernel will be used for every region. This is the fastest option. MKLMM with this option is equivalent to [Adaptive MultiBLUP](http://genome.cshlp.org/content/early/2014/06/24/gr.169375.113.abstract).
* rbf - An RBF kernel for every region
* poly2 - An inhomogeneous polynomial kernel of degree 2 for every region
* poly3 - An inhomogeneous polynomial kernel of degree 3 for every region
* nn - a neural network kernel (referred to in the paper as an SP kernel)  for every region
* adapt - MKLMM-Adapt: The code will adaptively select a different kernel for every region in a data driven manner. Note that this results in a slower run-time compared to the other kernels, because several kernels are evaluated for every region.

All kernels can be augmented with a linear kernel, by adding the suffix "_lin". For example, rbf_lin specifies that for every region the model will use a weighted combination of a linear and an RBF kernel. Note that the polynomial kernels become homogeneous when used along with a linear kernel, because otherwise the model is overparameterized. Also note that the code supports several other kernel types not reported in the paper, such as the Matern, Gabor and piecewise polynomial kernel. Please look at the source code for the full
list of kernels.

####Performing Prediction:
To perform prediction, one should type:
```
python mklmm_wrapper.py --bfile_train <bfile> --pheno_train <phenotype file> --covar_train <covariates file> --regions <regions file> --numRegions <#regions to use> --train_file <training file> --kernel <kernel type> --bfile <bfile with new individuals> --out <output file>
```

The bfile_train, pheno_train, covar_train, regions numRegions parameters should be exactly the same as in the previous run of mklmm_wrapper.py. The train_file is the output of the previous run of mklmm_wrapper.py
The file specified with --bfile should contain new individuals that did not participate in the training set. Prediction will be performed for these individuals.
The output is written to the output file. For every individual, the file reports the mean (column 3) and variance (column 4) of the posterior phenotype distribution for this individual.


The training set should be provided in the testing stage for several reasons:
* The SNPs of the test individuals should be normalized in the same way as the train individuals. It is possible to save the mean and standard deviation of every SNP, but this is wasteful, so was not used here.
* In order to compute the covariance between train and test individuals, the train genotypes are used. Note that one can bypass this requirement, as explained in the MKLMM paper, but this option is not yet enabled.
* Efficient computation of the predictive distribution of test individuals also requires the phenotypes and covariates of training individuals. Again, this requirement can be bypassed, but is not yet enabled.

Note that both runs of mklmm_wrapper.py can be combined. This is especially useful when one wants to remove principal components, as explained below.


------------------
Removal of top principal components
------------------
It is often beneficial to project genotype vectors into a subspace that is orthogonal to the top principal components of the data, in order to prevent confounding due to population structure. This can be done by using the flag "--numRemovePCs <#PCs>", and specifying a number larger than zero. Note that this should be done for both rank_regions.py and mklmm_wrapper.py to obtain consistent results.
When removing principal components, the test individuals should be provided (using the --bfile flag), so that the principal components computation will also consider their genotypes. 


-----------------------------
Standardization of genotypes
-----------------------------
Standardization of genotypes can be carried in several ways. The standard way is to simply standardize all SNPs to have a zero mean and a unit variance. Alternatively, one can apply an "ascertainment-aware" standardization by computing a weighted mean and standard deviations according to the disease prevalence, so that cases are not overrepresented in the computation. This can be accomplished with the flag "--norm controls". Note that unlike genotypes, standardization of covariates does not affect the analysis in any way, because they are treated as fixed effects.


-----------------------------
Example data set
-----------------------------
The package contains a small example dataset in the "example" directory with synthetic genotypes and phenotypes. The example contains two regions with interacting SNPs, as well as a polygenic term spanning all SNPs.
The data is divided into a train set (containing 1,500 individuals) and a test set (containing 1,301 individuals).
The example directory also contains the original (synthetic) phenotypes of all individuals, in the file pheno_all.phe, which can be used to evaluate prediction performance.
















