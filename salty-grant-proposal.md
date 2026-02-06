# **Project SALTY: A Multi-Modal Framework for High-Precision Geospatial Localization**

**Principal Investigator:** \[Your Name\]  
**Project Title:** Street-view Attention Learning Telemetry (SALTY)  
**Submission Date:** October 2025  
**Domain:** Computer Vision / Geospatial Machine Learning / AI Safety & Interpretability

## **1\. Project Summary / Abstract**

The field of planet-scale image geolocalization has seen rapid advancement with the introduction of foundation models such as PIGEON and GeoCLIP. These systems utilize massive Vision Transformers (ViTs) to retrieve coordinates from street-level imagery on a global scale. However, despite their impressive scope, these generalist models suffer from an inherent "curse of breadth." Current State-of-the-Art (SOTA) benchmarks indicate a median error of approximately 44km globally. While remarkable for identifying continents or countries, this resolution is functionally inadequate for hyper-local applications requiring neighborhood-level or street-level precision (e.g., autonomous last-mile delivery, rapid search and rescue, or verification of open-source intelligence).  
This project proposes **SALTY (Street-view Attention Learning Telemetry)**, a specialized, multi-modal transformer framework designed to solve the "last-mile" problem in geolocation. By restricting the geographic domain to the complex and heterogeneous biomes of California and employing a dense sampling strategy (\~50x higher density per square mile than global benchmarks), SALTY aims to achieve **sub-5km median accuracy**. The system leverages a Transfer Learning architecture, utilizing a frozen **CLIP ViT-L/14** backbone to extract robust semantic features, coupled with a custom, trainable geocell classification head. Crucially, SALTY introduces a "Glass Box" paradigm to address the critical interpretability deficit in current black-box AI. By integrating Gradient-weighted Class Activation Mapping (Grad-CAM) and a secondary Optical Character Recognition (OCR) fusion branch, SALTY transitions geolocation from an opaque probabilistic guess to a reasoned, evidence-based determination, mimicking human expert navigation.

## **2\. Specific Aims**

The proposed research seeks to bridge the critical gap between global information retrieval and high-precision local localization through three targeted scientific aims:  
**Aim 1: Quantify the Efficacy of Domain Specialization on Localization Accuracy**  
To demonstrate that restricting the problem space to a single, diverse state (California) allows for the implementation of significantly higher-resolution classification grids (**S2 Level 12/13**, \~1-3km) compared to the coarse grids required by global models (Level 9/10, \~25km). We hypothesize that a "Specialist" model trained on high-density data will reduce median error rates by an order of magnitude compared to "Generalist" baselines.  
**Aim 2: Develop an Interpretable "Glass Box" Geolocation Architecture**  
To mitigate the inherent opacity of deep neural networks in high-stakes tasks. This aim focuses on the integration of **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visualize the specific pixel regions (e.g., vegetation patterns, road markings) driving the model's predictions. Furthermore, we will interface the system with a Vision-Language Model (VLM) to generate natural language justifications for location estimates, fostering trust and human-AI collaboration.  
**Aim 3: Integrate Semantic Literacy into Geospatial Analysis**  
To move beyond purely visual pattern matching by implementing a multi-modal fusion branch. Current models treat text on street signs as abstract textures. SALTY will utilize **OCR (Optical Character Recognition)** to extract semantic text, process it via a lightweight language model (DistilBERT), and fuse these embeddings with visual features. This aims to replicate the human capability of explicit navigation (reading signs) rather than relying solely on implicit environmental cues.

## **3\. Background & Significance**

### **3.1 The Problem: The Capacity-Precision Tradeoff**

Current foundation models for geolocation optimize for maximizing global coverage. To keep the output space computationally feasible, they must discretize the earth into relatively large buckets (Geocells). PIGEON, for instance, must distribute its neural capacity across diverse environments ranging from the tundra of Siberia to the streets of Bangkok. This results in "catastrophic dilution," where the model learns broad strokes but fails to encode the subtle, high-frequency details necessary to distinguish neighboring cities or distinct micro-climates within the same region.

### **3.2 The Proposed Solution: SALTY's "Specialist" Hypothesis**

SALTY posits that deep learning models exhibit superior performance when the "Search Space" is constrained and the "Data Density" is increased. California serves as an ideal testbed for this hypothesis due to its extreme visual heterogeneity—encompassing distinct biomes such as coastal rainforests, alpine mountains, arid deserts, and dense urban metropolises. By training on a dataset with significantly higher density, SALTY forces the model to learn fine-grained discriminative features (e.g., specific species of oak trees or county-specific traffic signage) that global models ignore as noise.

## **4\. Proposed Methodology**

### **4.1 Architecture Design: Efficient Transfer Learning**

The system is architected to be computationally efficient while maximizing semantic understanding, utilizing a hybrid frozen-backbone approach:

* **Frozen Backbone (CLIP ViT-L/14):** We employ the **Contrastive Language-Image Pre-training (CLIP)** model as the primary feature extractor. CLIP's pre-training on 400 million image-text pairs provides robust, zero-shot recognition of environmental concepts. We freeze the weights of the ViT-L/14 to prevent "catastrophic forgetting" and to reduce the computational cost of training, focusing optimization solely on the downstream task.  
* **Geocell Classification Head:** The trainable component is a custom Multi-Layer Perceptron (MLP) head. This head projects the 768-dimensional CLIP embedding vector onto a probability distribution over **S2 Geometry Cells**.  
  * **S2 Geometry:** We utilize Google's S2 spherical geometry library to discretize California into hierarchical cells. We target **Level 12 (approx. 3km²)** granularity. This requires a classification output layer of approximately **2,000–5,000 active classes**, which is computationally trivial compared to the 50,000+ classes required for global coverage.  
* **OCR Fusion (Phase 3):** To address "Literacy," a secondary branch uses EasyOCR to extract scene text. Valid strings are tokenized and passed through a frozen **DistilBERT** model. The resulting text embeddings are concatenated with the visual CLIP embeddings via a Fusion Layer before entering the final classifier, allowing the model to override visual ambiguity with explicit textual evidence (e.g., "San Diego Fwy").

### **4.2 Data Acquisition & Rigorous Sampling Strategy**

Scientific reproducibility is central to the data engineering pipeline:

* **Source Data:** Official Google Street View 360° panoramas (Equirectangular projection), ensuring standardized camera height and optics.  
* **Reproducible Sampling Protocol:** From a superset of 1.5 million valid California coordinates, we generate a **uniform random sample of 100,000 training points** using a fixed cryptographic seed. This ensures the dataset is statistically representative of the state's geography and can be perfectly reconstructed by external auditors.  
* **Automated Quality Control:** A metadata filtering pipeline analyzes the copyright field of every panorama. The system automatically rejects user-uploaded content ("Photospheres") and indoor imagery, ensuring the model trains exclusively on high-quality, standardized outdoor street imagery.  
* **Resolution Optimization:** Data is acquired at **Zoom Level 2 (\~1664x832 px)**. This resolution provides sufficient pixel density for human-readable signage while maintaining a manageable storage footprint (\~60GB), adhering to the constraints of local compute infrastructure.

## **5\. Intellectual Merit & Broader Impacts**

### **5.1 Intellectual Merit**

This research contributes to the Computer Vision domain by:

1. **Quantifying the "Specialist Advantage":** Providing rigorous benchmarks comparing dense, region-specific models against sparse, global foundation models.  
2. **Advancing Multi-Modal Geolocation:** Moving beyond visual-only analysis to integrate semantic text (OCR) as a first-class feature in localization transformers.  
3. **Democratizing High-Performance AI:** Demonstrating that SOTA performance can be achieved on consumer-grade hardware (single GPU) through intelligent transfer learning and data curation, rather than massive compute clusters.

### **5.2 Broader Impacts**

The technology developed in SALTY has direct applications in critical public sectors:

* **Search and Rescue (SAR):** Rapidly geolocating photographs from lost hikers in California's vast wilderness areas (e.g., Sierra Nevada) where GPS data may be missing.  
* **Wildfire Analysis:** Enabling automated localization of crowd-sourced images of smoke columns or vegetation health to assist in early fire detection and forestry management.  
* **OSINT & Verification:** Providing a tool for journalists and researchers to verify the location of images circulating on social media, combating misinformation.  
* **Digital Archiving:** Assisting historians in identifying the locations of pre-GPS era photographs based on enduring landscape features.

## **6\. Facilities & Resources**

The project is supported by a dedicated, high-performance local compute node designed for efficient deep learning workflows:

* **Computational Hardware:** The "Windows Node" features an **AMD Ryzen 7 9800X3D CPU** for rapid data preprocessing, paired with an **NVIDIA RTX 5090 (32GB VRAM)**. This GPU provides the necessary memory bandwidth to batch-process high-resolution inputs and fine-tune large transformer heads.  
* **Software Environment:** A **Windows Native** stack managed via uv (pinned to **Python 3.10** for maximum stability) and Microsoft C++ Build Tools. This ensures strict compatibility with the latest PyTorch distributions and CUDA kernels.  
* **Development Workflow:** A decentralized **Git-based workflow** facilitates development on remote workstations (macOS) with synchronized execution on the central Windows GPU node, ensuring efficient code iteration and resource utilization.

## **7\. Project Timeline**

The research is structured into three distinct execution phases:

* **Month 1: Data Harvesting & Validation**  
  * Execution of the seeded 100k download script (salty\_scraper.py).  
  * Implementation of "Stealth" protocols (VPN rotation, randomized sleep) to ensure data continuity.  
  * Statistical analysis of the spatial distribution to verify coverage of all 58 California counties.  
* **Month 2: Baseline Model Development**  
  * Implementation of the custom PyTorch Dataset class with dynamic S2 cell mapping.  
  * Training and hyperparameter tuning of the Geocell Head on the frozen CLIP backbone.  
  * Benchmarking against "Naive" baselines (e.g., K-Nearest Neighbors on raw embeddings).  
* **Month 3: Advanced Features & Interpretability**  
  * Integration of Grad-CAM for interpretability analysis.  
  * Development and training of the OCR fusion branch.  
  * Final comparative analysis against PIGEON baselines using the "Median Error Distance" metric.