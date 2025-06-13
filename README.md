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

# SECURITY NOTE
 For access, use Hugging Face access token via environment variable, not hard-coded. See best practices in CONTRIBUTING.md


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
While prompt robustness in LLMs has been explored in prior literature, most efforts emphasize task-specific evaluations or isolated case analyses. In contrast, PBSS (Prompt-Based Semantic Shift) formalizes semantic-preserving rewordings as a structured axis of variation, and quantifies model sensitivity using standardized embedding-based diagnostics (e.g., SBERT).

To our knowledge, this is the first systematic effort to define PBSS as a protocol for evaluating token-level behavioral instability across LLMs. Unlike prior works which implicitly test prompt robustness, PBSS establishes a reproducible benchmark framework with interpretable output metrics and clearly defined perturbation dimensions.

This project does not claim isolation from related research—instead, it seeks to operationalize and consolidate this evaluation paradigm into an extensible toolset that others can adapt, build on, or critique.