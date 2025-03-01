# Unsupervised Alignment of Hypergraphs with Different Scales

**We are aware that the provided code, which is a best-restored version, has reproducibility issues. We are working on either reproducing the results or issuing a Corrigendum.**

Source code for the paper **Unsupervised Alignment of Hypergraphs with Different Scales**, where we formulate and address the problem of finding node correspondences across two hypergraphs, potentially with different scales, in an unsupervised manner.
To address the unique challenges of the problem, we propose **HyperAlign** (<ins><strong>Hyper</strong></ins>graph <ins><strong>Align</strong></ins>ment).
**HyperAlign** utilizes the hypergraph topology to extract node features (**HyperFeat**), conducts contrastive learning as an auxiliary alignment task (**HyperCL**), and employs Generative Adversarial Networks (GAN) to align the two respective node embedding spaces of the two hypergraphs. During the course of training GAN, **HyperAlign** augments each hypergraph with the "soft virtual hyperedges" from the counterpart hypergraph (**HyperAug**)) to resolve the scale disparity and share information across the two hypergraphs.
In a special case, we show that **HyperFeat** extracts node features as an implicit factorization of a constant matrix, is invariant up to any node permutation, and is expressive as a Hypergraph Isomorphism Test. In other words, in the most special case of hypergraph alignment that the two hypergraphs are actually isomorphic, i.e., they are different only by the node permutation, **HyperFeat** produces the two same sets of node embeddings.
Extensive experiments on ten real-world datasets demonstrate the significant and consistent superior of **HyperAlign** over the baseline approaches in terms of alignment performance.

If you have any questions, please feel free to contact Manh Tuan Do at: manh.it97@kaist.ac.kr


## Datasets
The datasets are in the *Datasets* zip. 

Source:
- coauth-Geology: https://www.cs.cornell.edu/~arb/data/coauth-MAG-Geology/
- coauth-History: https://www.cs.cornell.edu/~arb/data/coauth-MAG-History/
- contact-high: https://www.cs.cornell.edu/~arb/data/contact-high-school/
- contact-primary: https://www.cs.cornell.edu/~arb/data/contact-primary-school/
- email-Enron: https://www.cs.cornell.edu/~arb/data/email-Enron/
- email-Eu: https://www.cs.cornell.edu/~arb/data/email-Eu/
- NDC-classes: https://www.cs.cornell.edu/~arb/data/NDC-classes/
- NDC-substances: https://www.cs.cornell.edu/~arb/data/NDC-substances/
- threads-ask-ubuntu: https://www.cs.cornell.edu/~arb/data/threads-ask-ubuntu/
- threads-math: https://www.cs.cornell.edu/~arb/data/threads-math-sx/

## Requirements:
- Pytorch
- futures
- fastdtw
- gensim
  
## Code
The source code is in the *HyperAlign* folder.

## How to run the code:
starting at the *HyperAlign* folder, store the datasets in the *dataset* folder and run the command:

python main.py --dataset1 [NAME1] --dataset2 [NAME2] --iter [ITER] --input_dimensions [INPUT DIM] --hid_dim [HIDDEN DIM] --epochs [EPOCHS] --episode [EPISODE] --t [T] --config [OPTION] --pred [PRED]
- [NAME1]: the name of the file containing the list of hyperedges of the first hypergraph.
- [NAME2]: the name the file containing the list of hyperedges of the second hypergraph.
- [ITER]: the number of SGD iterations in language model.
- [INPUT DIM]: the dimension of the node features extracted in HyperFeat.
- [HIDDIM DIM]: dimension of the output node embeddings.
- [EPOCHS]: the number of epochs in contrastive learning
- [EPISODE]: the number of epochs for adversarial learning.
- [T]: the number of similar nodes, in the counter-part hypergraph, to construct augmented incidence matrice.
- [OPTION]: choose 0/1/2/3/4 (0 as default)
   > 0: full-fledged HyperAlign (default option),
   > 1: HyperAlign-s,
   > 2: HyperAlign-WC,
   > 3: HyperAlign-WA,
   > 4: HyperAlign-WAC
- [PRED] whether to save alignment prediction (by choosing the node having the most similar embedding) in the prediction output folder.
  > 0: to not save,
  > 1: to save

For example: python main.py --dataset1 toy1 --dataset2 toy2 --iter 30 --input_dimensions 32 --hid_dim 64 --epochs 20 --episode 30 --t 3 --pred 1

The output node embeddings will be stored in folder *embeddings*, and the prediction will be stored in folder *prediction*.
