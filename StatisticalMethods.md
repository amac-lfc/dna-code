# Statistical Methods

## Principal Component Analysis


	Input: set of possibly correlated variables used to predict a final variable
	Output: Seprated groups of those variables all linearly uncorrelated to one another
	Why: used to seperate variables into the best state to be used for a prediction

		Structure: an array of variables with the first object having the highest variance, each proceding object has the next highest variance while still being orthongal to the preceding 					variables

		Components per object: component score(the variables values corresponding to a specific data point) and the loadings(the weigh each origional variable should be multiplied by to get the 						component score)

	How to Do it:
		1) spectral decompisiton of a data covariance matrix (the thing they did in the original paper that you liked)
		   OR
		1) singular value decomposition of a design matrix(Each row represents an individual object, with the successive columns corresponding to the variables)
			1a) if done this way you need to normalize the data, normalization of the data through mean centering and normalizing each variables variance to 1 (z scores could be useful)


