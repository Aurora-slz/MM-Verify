# MM-Verifier: Enhancing Multimodal Reasoning with Chain-of-Thought Verification
[[Huggingface Dataset](https://huggingface.co/datasets/lhpku20010120/MM-Verify-Data/tree/main)] [[MM-Verifier](https://huggingface.co/lhpku20010120/MM-Verify/tree/main)]  [[Paper](https://arxiv.org/abs/2502.13383)]

Test-time scaling enables a model to generate more tokens during the inference stage, which is an effective approach to enhancing accuracy. Designing an effective verifier is key to significantly improving reasoning performance. However, in the multimodal (MM) domain, there is still a lack of a strong MM-Verifier. In this paper, we introduce MM-Verifier and MM-Reasoner to enhance multimodal reasoning through longer inference and more robust verification. First, we propose a two-step MM verification data synthesis method, which combines a simulation-based tree search with verification and uses rejection sampling to generate high-quality Chain-of-Thought (COT) data. This data is then used to fine-tune the verification model, MM-Verifier. Additionally, we present a more efficient method for synthesizing MMCOT data, bridging the gap between text-based and multimodal reasoning. The synthesized data is used to fine-tune MM-Reasoner.

> We will organize the code after the paper submission. Thank you for your attention!

## 💥 News 💥

- **[2025.02.23]** 💥 We released MM-Verifier model. [[MM-Verifier](https://huggingface.co/lhpku20010120/MM-Verify/tree/main)]
- **[2025.02.23]** 💥 We released training dataset of MM-Verifier and MM-Reasoner. [[Huggingface Dataset](https://huggingface.co/datasets/lhpku20010120/MM-Verify-Data/tree/main)]

## MM-Verify  
+ Search Algorithm  
`/search/eval.sh  `
> We referred to the awesome work [ResT-MCTS](https://github.com/THUDM/ReST-MCTS) for the implementation of the search algorithm, thanks!
  
+ Each question is sampled $n$ times  
`/data_syn/sample_qwen2vl.py  `

+ Perform ORM data annotation  
`/data_syn/orm_to_sft.py ` 

+ Data cleaning  
`/data_syn/clean_ormData_mm_sample.py  `

## MM-Reasoning  
+ Use QwQ for data distillation  
`/data_syn/test4_mavis_vllm_slz.py  `

+ Perform data cleaning  
`/data_syn/clean_qwqData_mm.py  `
