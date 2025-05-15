# Demo Code for No.9750 Submission

## Introduction
This is the demo code for our paper. The detailed version will be released once this paper is accepted.

## Running configuration
Our method consists of several steps. We provide the necessary commands in `script/`

1. Run `sh scripts/train_essential_model.sh `. This command helps you train a VAE and the score model required for computing score similarities.
2. Run `sh scripts/obtain_discrete_scores.sh `. This command will sample several timesteps with different noise scales. The checkpoint will be saved for ultimate similarity computing.
3. Run `bash scripts/train_classification.sh`. You will train the prediction model with our score-based reweighting method.
4. Run `bash scripts/test_classification.sh`. Test the performance on selected non-causal attributes to evaluate robustness.


## Acknowledgement
Our codebase for the diffusion models builds heavily on [TabSyn](https://github.com/amazon-science/tabsyn). The preprocess on Taxi and ACS dataset follows the setting in [WhyShift](https://github.com/namkoong-lab/whyshift/tree/main).

Thanks for open-sourcing!