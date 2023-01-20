# ood-detection

This project examines the problem of out-of-distribution (OOD) detection in machine learning, which is the situation where a network is exposed to data from classes outside of the distribution of classes seen during training. For example, a network trained to identify dog breeds will still attempt to predict the dog breed pictured in an image of a cat and output a confident prediction. In reality, this overly confident prediction should not be output, so being able to detect an OOD input and refuse to make this error is valuable for a model.  

jidf
