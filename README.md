# LLaVA³: Representing 3D Scenes like a Cubist Painter to Boost 3D Scene Understanding of VLMs


![Teaser](img/teaser.jpg)

Doriand Petit¹² · Steve Bourgeois¹ · Vincent Gay-Bellile¹ · Florian Chabot¹ · Loïc Barthe²

¹Université Paris-Saclay, CEA List · ²IRIT, Université Toulouse III, CNRS


## 💡 Abstract

Developing a multi-modal language model capable of understanding 3D scenes remains challenging due to the limited availability of 3D training data, in contrast to the abundance of 2D datasets used for vision-language models (VLM). As an alternative, we introduce LLaVA³ (pronounced LLaVA-Cube), a novel method that improves the 3D scene understanding capabilities of VLM using only multi-view 2D images and without any fine-tuning. Inspired by Cubist painters, who represented multiple viewpoints of a 3D object within a single picture, we propose to describe the 3D scene for the VLM through omnidirectional visual representations of each object. These representations are derived from an intermediate multi-view 3D reconstruction of the scene. Extensive experiments on 3D VQA and 3D language grounding show that our approach outperforms previous 2D-based VLM solutions.


🎉 Accepted at AAAI'26 — See you in Singapore ! 🎉

## 🔍 Overview

Our pipeline consists of three main stages:

    1. Multi-view 3D LLaVA and hierarchical SAM Reconstruction: Training a Nerfacto-based NeRF that jointly learns geometry, a dense 3D LLaVA feature field and a hierarchical Instance SAM-CLIP-based feature field.

    2. Object Hierarchy Extraction: Decomposing the scene into a graph of objects, parts, and sub-parts using clustering on our hierarchical feature field and refining this graph via CLIP-based heuristics filtering.

    3. VLM-compatible Representation: converting these 3D objects into ordered 2D "Cubist" token sets that a standard VLM can understand without fine-tuning.

This set of visual tokens can then be fed to the LLaVA VLM with any textual queries.



## 📄 Paper & Supplementary Material

    - 📄 Paper & Supplementary Material: Coming soon


## 🧾 Citation

If you find this project useful, please cite:

    @inproceedings{petit2024ring,
        title={RING-NeRF: Rethinking Inductive Biases for Versatile and Efficient Neural Fields},
        author={Petit, Doriand and Bourgeois, Steve and Pavel, Dumitru and Gay-Bellile, Vincent and Chabot, Florian and Barthe, Loic},
        journal={European Conference on Computer Vision (ECCV)},
        year={2024}
    }


## 🙏 Acknowledgements

This work was made possible thanks to the use of the CEA List FactoryIA supercomputer, supported by the Île-de-France Regional Council.

The website design was adapted from Michaël Gharbi, Ref-NeRF, nerfies.

## 📬 Contact

For questions or collaborations, reach out via GitHub.
