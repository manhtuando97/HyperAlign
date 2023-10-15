# Unsupervised Alignment of Hypergraphs with Different Scales
Source code for the paper **Unsupervised Alignment of Hypergraphs with Different Scales**, where we formulate and address the problem of finding node correspondences across two hypergraphs, potentially with different scales, in an unsupervised manner.
To address the unique challenges of the problem, we propose **HyperAlign** (<ins><strong>Hyper</strong></ins>graph <ins><strong>Align</strong></ins>ment).
**HyperAlign** utilizes the hypergraph topology to extract node features (**HyperFeat**), conducts contrastive learning as an auxiliary alignment task (**HyperCL**), and employs Generative Adversarial Networks (GAN) to align the two respective node embedding spaces of the two hypergraphs. During the course of training GAN, **HyperAlign** augments each hypergraph with the "soft virtual hyperedges" from the counterpart hypergraph (**HyperAug**)) to resolve the scale disparity and share information across the two hypergraphs.
In a special case, we show that **HyperFeat** extracts node features as an implicit factorization of a constant matrix, is invariant up to any node permutation, and is expressive as a Hypergraph Isomorphism Test. In other words, in the most special case of hypergraph alignment that the two hypergraphs are actually isomorphic, i.e., they are different only by the node permutation, **HyperFeat** produces the two same sets of node embeddings.
Extensive experiments on ten real-world datasets demonstrate the significant and consistent superior of **HyperAlign** over the baseline approaches in terms of alignment performance.


## Datasets
The datasets are in the *Datasets* folder. 
We remove any nodes that are incident to fewer than two hypergraphs.

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


## Code
The source code is in the *src* folder.

## How to run the code:
starting at the *src* folder, run the command:

*python main.py --dataset_1 --dataset 2
- dataset_2: the file containing the list of hyperedges of the first hypergraph.
- dataset_2: the file containing the list of hyperedges of the second hypergraph.

For example: 
