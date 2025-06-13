# LLM-Prompt-Variance-Diagnostic-Analysis

# PBSS Toolkit
This repository contains the toolkit, evaluation code, and visualizations for the paper:
"When Meaning Stays the Same, but Models Drift: Evaluating Quality of Service under Token-Level Behavioral Instability in LLMs"

PBSS defines semantic-preserving prompt variation and evaluates model behavior stability under surface-level changes. While this version benchmarks English prompts, the structure is extensible to multilingual domains. We encourage further extensions using the same evaluation framework.

This is a preliminary version of ongoing work exploring behavioral instability in LLMs under semantic consistency.

We plan to expand the evaluation framework in future updates.


# Custom Prompt Variants
While this repository includes a predefined set of semantically equivalent prompt variants for evaluation, developers in the community are welcome to construct their own variant sets.

As long as the new prompt variants satisfy the semantic equivalence criteria and pass the sanity check procedures outlined in our paper (Section 4), the PBSS framework can process and evaluate them reliably.

This flexibility enables broader experimentation with different prompt templates, user domains, or task settings.



# Compliance and Ethics Statement
This repository does **not** include or rely on any private, sensitive, or institutionally-restricted data.

All prompts, embeddings, and model outputs used in the PBSS benchmark are derived from publicly available APIs or open model endpoints. No internal system access or privileged datasets were used at any point.

The experimental process was conducted independently and in full accordance with standard academic integrity practices and the institutional Honor Code.

All data generation and evaluation processes were manually audited and documented. This benchmark is intended solely for academic transparency and research reproducibility.

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

# Q&A: Note on Related Work
While the robustness of LLMs to prompt variation has been explored in prior research, existing approaches often focus on task-specific metrics or qualitative case studies. PBSS, by contrast, frames semantic-preserving perturbations as a controlled axis of variation and applies standardized embedding-based evaluation (e.g., SBERT) across multiple models and decoding settings.
To our knowledge, this is the first effort to define and benchmark Prompt-Based Semantic Shift (PBSS) as a general-purpose protocol for measuring behavioral instability in LLMs.
Our work does not claim to be isolated from community efforts—but rather aims to systematize and operationalize this evaluation space under a reproducible, extensible framework.

