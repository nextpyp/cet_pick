# Cellular content exploration and accurate particle localization

MiLoPYP is a dataset-specific contrastive learning-based framework that enables two-step fast molecular pattern visualization followed by accurate protein localization without the need for manual annotation. During the exploration step, it learns an embedding space for 3D macromolecules such that similar structures are grouped together while dissimilar ones are separated. The embedding space is then projected into 2D and 3D which allows easy identification of the distribution of macromolecular structures across an entire dataset. During the refinement step, examples of proteins identified during the exploration step are selected and MiLoPYP learns to localize these proteins with high accuracy.

MiLoPYP can be run as a standalone application or through the [nextPYP](https://nextpyp.app) suite.

For further information, read the [documentation](https://nextpyp.app/milopyp).

The software is distributed open source under the [BSD 3-Clause license](../LICENSE).