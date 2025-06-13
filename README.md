# LLM-Prompt-Variance-Diagnostic-Analysis

# PBSS Toolkit
This repository contains the toolkit, evaluation code, and visualizations for the paper:
"When Meaning Stays the Same, but Models Drift: Evaluating Quality of Service under Token-Level Behavioral Instability in LLMs"

This is a preliminary version of ongoing work exploring behavioral instability in LLMs under semantic consistency.

We plan to expand the evaluation framework in future updates.


# Custom Prompt Variants
While this repository includes a predefined set of semantically equivalent prompt variants for evaluation, developers in the community are welcome to construct their own variant sets.

As long as the new prompt variants satisfy the semantic equivalence criteria and pass the sanity check procedures outlined in our paper (Section 4), the PBSS framework can process and evaluate them reliably.

This flexibility enables broader experimentation with different prompt templates, user domains, or task settings.


# Citation
If you use the PBSS framework or any part of this repository in your research, please cite the following paper:
<pre>
@misc{li2025meaningstayssamemodels,
      title={When Meaning Stays the Same, but Models Drift: Evaluating Quality of Service under Token-Level Behavioral Instability in LLMs}, 
      author={Xiao Li and Joel Kreuzwieser and Alan Peters},
      year={2025},
      eprint={2506.10095},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.10095}, 
}</pre>

