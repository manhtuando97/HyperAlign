This folder contains the additional results for the rebuttal. Due to the the space constraint, we could not upload all the results directly in the rebuttal. Please refer to each file for the additional experimental results that we have performed.

1. Experimental results when the disparity scale between the two input hypergraphs is varied.
 (in response to **W3** by **Reviewer NPqf** and **W2** by **Reviewer 6tyM**)

- The methods considered here are: **HyperAlign**, **HyperAlign-WA**, **HyperAlign-WC**, **HyperAlign-WAC** (the variants of **HyperAlign**), and **WAlign**, the strongest competitor.
- These results validate the contributions of the two modules: Contrastive Learning and Topological Augmentation to the performance when the scale disparity between the two input hypergraphs increases. 
  - File 'varying_Configuration_1.png' corresponds to **Configuration 1** and supplements **Figure 2** of the main paper.
  - File 'varying_Configuration_2.png' corresponds to **Configuration 2** and supplements **Figure 4** of the main paper.

2. Experimental results when recovering the self-loops: to verify that including self-loops does not contribute to any noticeable performance difference.
 (in response to **W3** by **Reviewer 8Fky**)
- Comparing performance of **HyperAlign** with self-loops, **HyperAlign-self**, and HyperAlign in the datasets after self-loops have been removed:

  - File 'Configuration_1_self-loops.JPG': for **Configuration 1** 
  - File 'Configuration_2_self-loops.JPG': for **Configuration 2** 

3. Experimental results of two additional competitors:
   
 (in response to **W1** by **Reviewer 6tyM**)
 
- SANA [1]
  
[1] Peng J, Xiong F, Pan S, et al. Robust Network Alignment with the Combination of Structure and Attribute Embeddings. IEEE International Conference on Data Mining (ICDM) 2023. 

- Grad-Align+ [2]
  
[2] Park J D, Tran C, Shin W Y, et al. GradAlign+: Empowering gradual network alignment using attribute augmentation. ACM CIKM 2022.

- Comparison of **SANA**, **Grad-Align+**, and **HyperAlign**.
  - File 'Configuration_2_baselines.JPG': for **Configuration 2**.
   (The results for Configuration 1 have been included in our response to W1).

