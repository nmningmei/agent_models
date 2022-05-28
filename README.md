# The simulation modeling part of [Mei, N. Santana, R., & Soto, D. (2022). Informative neural representations of unseen objects during higher-order processing in human brains and deep artificial networks, Nature Human Behavior. https://doi.org/10.1038/s41562-021-01274-7](https://github.com/nmningmei/unconfeats)

# Requirement

- python==3.6.3
- pytorch==1.11.0
- tensorflow==2.0.1
- scikit-learn==1.0

# Main results
## When we train the model configurations without noise (adding Gaussian noise to the image)
![performance](https://github.com/nmningmei/agent_models/blob/master/figures/CNN_performance.jpeg)

![decode](https://github.com/nmningmei/agent_models/blob/master/figures/decoding_performance.jpeg)

![featureimportance](https://github.com/nmningmei/agent_models/blob/master/figures/feature%20importance.jpeg)
## When we train the model confirations with standard normal noise embedded

![cnn_noise](https://github.com/nmningmei/agent_models/blob/master/figures/trained%20with%20noise%20performance.jpg)

![cnn_chance](https://github.com/nmningmei/agent_models/blob/master/figures/trained%20with%20noise%20chance%20cnn.jpg)
