# Rebuilding a 2018 "Saree Design Intelligence System" from Verified Real Resources

## TL;DR
- Nine of the ten blueprint layers map cleanly onto **real, verifiable resources that existed in 2018 or earlier** — Electron docs, SQLite, official OpenCV-Python tutorials, PyImageSearch tutorials by Adrian Rosebrock, scikit-learn/scikit-image examples, the Keras blog and Applications docs, TensorFlow for Poets, and FAISS (open-sourced by Facebook AI Research in early 2017). Chart.js and the Node↔Python bridge (python-shell) also predate 2018.
- The **only layer without a clean 2018 saree-specific resource** is the domain logic itself: the "saree pattern taxonomy" (floral/paisley/temple/butta) is bespoke, and one Indian "saree/handloom" paper commonly cited turns out to be a garbled citation. Verified 2018-or-earlier substitutes exist for adjacent domains (batik retrieval, woven-fabric SVM, DeepFashion).
- A realistic 2018 builder's research trail runs **UI → storage → classical CV → classical ML → deep features → similarity search → analytics → glue**, and every step can be sourced to a named author/org with a datable URL — no fabrication required.

## Key Findings
- **Every core technical layer is genuinely reconstructable from 2018.** The classical-CV and ML stack was mature by 2015; deep-learning transfer learning was well documented by 2016–2017; FAISS shipped in 2017; Electron/Chart.js/python-shell were all established.
- **PyImageSearch (Adrian Rosebrock) is the single most load-bearing source**, supplying datable tutorials for color k-means (2014), the image-search-engine series (2014), Local Binary Patterns (2015), shape/contour/color (2016), and Keras image classification (2017–2018).
- **The saree-specific pattern categories are the invented part of the blueprint** — no single 2018 paper defines "temple/butta/paisley" classification. The closest verifiable 2018-or-earlier textile work is on batik and woven-fabric texture classification, plus the DeepFashion dataset (2016).
- **One frequently-repeated "Praveen Kumar 2018 handloom fabric" citation is unreliable** and should not be used; its described method actually belongs to a fleece-pilling paper (Huang & Fu 2018, *Fibers* 6(4):73), which reports "Classification accuracies of the ANN and SVM were 96.6% and 95.3%, respectively" on 320 fleece samples graded 2–5 (80 per grade).

## Details — Layer-by-Layer Mapping

### Layer 1 — Electron desktop UI
- **electron-quick-start** (GitHub, `electron/electron-quick-start`) — the official minimal Electron starter app (package.json, main.js, index.html), maintained by the Electron org; it long predates 2018 and is the canonical first thing a builder clones. URL: https://github.com/electron/electron-quick-start
- **Electron official documentation / Quick Start Guide** (electronjs.org, formerly electron.atom.io/docs). The 2018-era docs were hosted at electron.atom.io before migrating to electronjs.org.
- *Maps to saree logic:* provides the main-process/renderer-process window that hosts the shop's catalog browser and the "upload a saree photo → see similar designs" screen.

### Layer 2 — SQLite image/inventory storage
- **SQLite official documentation** (sqlite.org) — the canonical reference; SQLite has been documented there since the early 2000s.
- **Python `sqlite3` module docs** (docs.python.org) — part of the Python standard library since Python 2.5, so available to any 2018 Python 3.x build.
- **`node-sqlite3`** (npm / `mapbox/node-sqlite3` on GitHub) — the established Node binding used from Electron's main process; predates 2018.
- *Maps to saree logic:* stores saree records (SKU, fabric, price, region), the image blobs/paths, and the extracted feature vectors and dominant-color histograms.

### Layer 3 — OpenCV image preprocessing (resize, grayscale, threshold, Canny, contours)
- **Official OpenCV-Python Tutorials** at docs.opencv.org (and the widely-mirrored OpenCV 2.4 tutorials on readthedocs). Verified individual pages that existed pre-2018:
  - **"Canny Edge Detection"** — OpenCV-Python Tutorials (documents `cv2.Canny()`, Canny's 1986 algorithm, hysteresis thresholds).
  - **"Contours: Getting Started"** — documents `cv2.findContours()` and the advice to apply threshold/Canny first ("before finding contours, apply threshold or canny edge detection").
  - Image thresholding, changing colorspaces, and histogram tutorials in the same `py_imgproc` tree.
- *Maps to saree logic:* normalizes photos taken in-shop (resize to a common size, grayscale + threshold to isolate the pallu/border, Canny + contours to find repeated motif boundaries).

### Layer 4 — Pattern recognition (OpenCV features + scikit-learn RF/SVM)
- **scikit-learn official examples** — e.g. "OOB Errors for Random Forests," "Plot the decision surfaces of ensembles of trees on the iris dataset," "Feature importances with a forest of trees," and the `RandomForestClassifier`/SVM API docs at scikit-learn.org. These example pages date to the 2014-era docs and earlier.
- **PyImageSearch "Local Binary Patterns with Python & OpenCV"** (Adrian Rosebrock, Dec 7 2015) — demonstrates exactly the "extract features → train a Linear SVM → classify texture/pattern" pipeline this layer needs. URL: https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
- *Maps to saree logic:* the 8-way saree pattern classifier (floral, paisley, temple, checks, stripes, geometric, butta, abstract) is a Random Forest / Linear SVM trained on hand-engineered features — a direct application of the LBP→SVM recipe with saree-specific class labels.

### Layer 5 — Colour intelligence (RGB/HSV, histograms, k-means dominant colours)
- **PyImageSearch "OpenCV and Python K-Means Color Clustering"** (Adrian Rosebrock, May 26 2014) — uses `sklearn.cluster.KMeans` on reshaped pixel arrays to find dominant colors; the canonical dominant-color recipe. URL: https://pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
- **PyImageSearch "Color Quantization with OpenCV using K-Means Clustering"** (Jul 7 2014) — companion piece for reducing palettes.
- **PyImageSearch "Determining object color with OpenCV"** (Feb 15 2016) and the shape/contour series ("OpenCV center of contour," Feb 1 2016; "OpenCV shape detection," Feb 8 2016) — L*a*b* color labeling on masked contour regions.
- **Official OpenCV-Python histogram / changing-colorspaces tutorials** (docs.opencv.org).
- *Maps to saree logic:* extracts the 3–5 dominant colors of each saree (for "show me red temple-border silks") and builds color histograms as part of the design fingerprint.

### Layer 6 — Texture analysis (scikit-image LBP, Gabor)
- **scikit-image "Gabor filter banks for texture classification"** gallery example (scikit-image.org, `plot_gabor`) — computes mean/variance of Gabor-filtered responses as texture features; the example exists in scikit-image docs going back to v0.9/v0.11/v0.12 (well before 2018).
- **scikit-image `local_binary_pattern`** API + the PyImageSearch LBP post (2015) referenced above, which explicitly recommends the scikit-image LBP implementation ("I recommend using the scikit-image implementation of LBPs").
- *Maps to saree logic:* distinguishes weave/texture families (silk vs. cotton vs. brocade) and fine motif texture (butta density) that pure color/shape misses.

### Layer 7 — Deep learning feature extraction (Keras transfer learning; VGG16/ResNet50/MobileNet)
- **Keras Applications documentation** (keras.io/api/applications, and the 2017-era mirror at faroit.com/keras-docs/2.0.x and 2.1.x) — documents `VGG16`, `ResNet50`, `MobileNet` with `include_top=False` for feature extraction and the `pooling` argument. MobileNet appears from Keras 2.0.6/2.0.8 (2017).
- **Keras blog: "Building powerful image classification models using very little data"** (François Chollet) — the seminal transfer-learning-with-VGG16 / bottleneck-features tutorial. The page's own metadata dates it to **2016-06-05 (Sun 05 June 2016), "By Francois Chollet,"** and it notes it was originally written in June 2016. URL: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
- **PyImageSearch "Image classification with Keras and deep learning"** (Dec 11 2017) and **"Keras Tutorial: How to get started with Keras, Deep Learning, and Python"** (Sep 10 2018) — practitioner guides to training a CNN on a custom image dataset.
- *(Note: Rosebrock's dedicated "Transfer Learning with Keras" feature-extraction series — "Transfer Learning with Keras and Deep Learning," "Keras: Feature extraction on large datasets," "Fine-tuning with Keras" — is May–June 2019, just after the 2018 cutoff, so the 2016 Keras blog post and the Keras Applications docs are the correct 2018-era sources for this layer.)*
- **TensorFlow for Poets** (Pete Warden, Feb 28 2016; official Google Codelab `googlecodelabs/tensorflow-for-poets-2`) — retrains the final layer of a pretrained ImageNet CNN on a custom image set; a real 2016 alternative path for transfer learning.
- *Maps to saree logic:* the "design fingerprint" is the flattened/pooled feature vector from a pretrained CNN with its classifier head removed — a robust embedding for every saree image.

### Layer 8 — Similar-design search (FAISS)
- **FAISS** (`facebookresearch/faiss`, GitHub) — open-sourced by Facebook AI Research; original authors Hervé Jégou, Matthijs Douze, Jeff Johnson; initial public release February 2017.
- **Facebook Engineering blog: "Faiss: A library for efficient similarity search"** (March 29 2017), which states verbatim: *"This month, we released Facebook AI Similarity Search (Faiss)… We've built nearest-neighbor search implementations for billion-scale data sets that are some 8.5x faster than the previous reported state-of-the-art, along with the fastest k-selection algorithm on the GPU known in the literature."* URL: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
- **TechCrunch coverage** (March 29 2017) corroborates the release date.
- *Maps to saree logic:* indexes all saree fingerprint vectors and returns nearest neighbours for "find sarees that look like this one" in milliseconds.

### Layer 9 — Sales correlation / prediction (scikit-learn regression/classification)
- **scikit-learn official documentation and examples** (scikit-learn.org) — regression and classification estimators with example galleries; mature well before 2018.
- *Maps to saree logic:* correlates visual attributes (pattern class, dominant color, texture) and price/region with historical sales to predict which incoming designs will sell.

### Layer 10 — Recommendation engine (visual similarity) + Chart.js + Node↔Python bridge
- **Recommendation via visual similarity** reuses Layer 7 (Keras embeddings) + Layer 8 (FAISS). Conceptual grounding: the **PyImageSearch "Building an Image Search Engine" series** (Adrian Rosebrock, Feb 2014: Step 1 Feb 3, Step 2 Feb 10, Step 3 Feb 17, Step 4 Feb 24; plus "Hobbits and Histograms," Jan 27 2014, and "The complete guide to building an image search engine with Python and OpenCV," Dec 1 2014). These define the descriptor→index→similarity-metric→search framework.
- **Chart.js documentation** (chartjs.org) — Chart.js 2.x docs (versions 2.7.3 and 2.9.4 are archived) available in 2018.
- **python-shell** (npm `python-shell`, `extrabacon/python-shell` on GitHub) — runs Python scripts from Node via stdio; established well before 2018. Alternative: Node.js `child_process` core docs.
- *Maps to saree logic:* Chart.js renders the sales/inventory dashboards inside the Electron window; python-shell lets the Electron/Node front end call the Python CV/ML scripts.

### Domain grounding — saree/textile classification research (the invented layer)
- **DeepFashion dataset** — Liu, Luo, Qiu, Wang, Tang, CVPR 2016 (pp. 1096–1104), "DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations." Per the CVF Open Access paper, DeepFashion *"contains over 800,000 images, which are richly annotated with massive attributes, clothing landmarks, and correspondence of images taken under different scenarios including store, street snapshot, and consumer."* This is the strongest real anchor for "clothing/fashion recognition and retrieval." URL: https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html
- **Batik retrieval with Gabor features** — Prasetyo H., Wiranto W., Winarno W., "Statistical Modeling of Gabor Filtered Magnitude for Batik Image Retrieval," *Journal of Telecommunication, Electronic and Computer Engineering (JTEC)*, Vol. 10, No. 2-4, pp. 85–89, 2018; hosted at jtec.utem.edu.my/jtec/article/view/4322. A verified 2018 traditional-textile (batik) retrieval paper using Gabor-magnitude features.
- **Woven-fabric texture classification with SVM** — Ben Salem, Y. & Nasri, S., "Automatic recognition of woven fabrics based on texture and using SVM," *Signal, Image and Video Processing (SIViP)* 4, 429–434 (2010), DOI 10.1007/s11760-009-0132-5. The paper reports: *"Although it is the oldest method, GLCM always remains accurate (97.2%). The fusion of the Gabor wavelet and GLCM gives the best result (98%)"* using an SVM to classify twill/satin/plain weaves. Real, but 2010 — usable as prior art, published before the 2018 cutoff.
- **FLAGGED — do not use:** the widely-repeated "Praveen Kumar et al. 2018, handloom fabric identification with Gabor + LBP, SVM/ANN, grades 2–5" citation is unreliable. Tracing it back, the described method/results actually belong to Huang, M.-L. & Fu, C.-C., "Applying Image Processing to the Textile Grading of Fleece Based on Pilling Assessment," *Fibers* 2018, 6(4):73, DOI 10.3390/fib6040073 — 320 fleece samples graded 2–5 (80 per grade), with classification accuracies of 96.6% (ANN) and 95.3% (SVM). That is a fleece-pilling paper, not a handloom/saree paper. No genuine "saree pattern taxonomy" paper from 2018 could be verified.

## Realistic 2018 Research Trail (first read → last)
1. **electron-quick-start** + Electron docs — stand up the desktop shell.
2. **SQLite docs / Python sqlite3 / node-sqlite3** — design the catalog schema.
3. **Official OpenCV-Python tutorials** (colorspaces, thresholding, Canny, contours, histograms) — learn preprocessing.
4. **PyImageSearch K-Means Color Clustering (2014)** and **Determining object color (2016)** — dominant-color extraction.
5. **PyImageSearch Local Binary Patterns (2015)** + **scikit-image Gabor example** — texture features.
6. **scikit-learn RF/SVM examples** — train the pattern classifier.
7. **Keras blog "…very little data" (June 2016)** + **Keras Applications docs** + **PyImageSearch Keras tutorials (2017–2018)** / **TensorFlow for Poets (2016)** — deep feature extraction / design fingerprints.
8. **FAISS GitHub + Facebook Engineering announcement (March 2017)** — similarity search.
9. **PyImageSearch "Building an Image Search Engine" series (2014)** — tie descriptors + similarity into a recommender.
10. **scikit-learn regression docs** — sales correlation/prediction.
11. **Chart.js docs** + **python-shell** — dashboards and the Node↔Python bridge.
12. **DeepFashion (2016)** + batik (JTEC 2018) / woven-fabric (SIViP 2010) texture papers — domain grounding for the saree taxonomy.

## Recommendations
- **Build in the trail order above.** Ship a thin vertical slice first: Electron shell + SQLite + one OpenCV preprocessing step + K-means dominant color, so the shop sees value before the deep-learning work.
- **Use classical features (LBP + Gabor + color histograms → SVM/RF) as the v1 pattern classifier.** They are fully 2018-sourced, cheap to train on a small hand-labeled saree set, and interpretable. Reserve CNN embeddings + FAISS for the "find similar" feature where they add the most.
- **Treat the saree taxonomy as bespoke, hand-labeled data, not a cited model.** Since no 2018 paper defines temple/butta/paisley classes, budget for building a labeled dataset; borrow DeepFashion's annotation methodology (50 categories, 1,000 attributes, landmarks) as a template.
- **Drop the "Praveen Kumar 2018 handloom" reference entirely** and cite the batik (JTEC 2018) and woven-fabric (SIViP 2010) papers plus DeepFashion (CVPR 2016) for domain credibility.
- **Benchmarks that would change the plan:** if the classical SVM pattern classifier exceeds ~85–90% on your labeled saree set, you may not need CNN classification at all (keep the CNN only for embeddings). If FAISS latency is irrelevant at your catalog size (a few thousand sarees), a plain scikit-learn `NearestNeighbors` brute-force search is sufficient and simpler — reserve FAISS for catalogs in the tens of thousands.

## Caveats
- **Publication-date confidence.** PyImageSearch, the Keras blog, Facebook Engineering, CVPR, the JTEC/SIViP/Fibers papers, and versioned Keras/Chart.js/scikit-image docs carry explicit dates and are high-confidence pre-2018 (or explicitly dated within 2018). Some pages (PyImageSearch, OpenCV, scikit-learn, Keras Applications) have been **updated/re-hosted since 2018** (e.g., "last updated 2026" banners; electron.atom.io → electronjs.org; blog.keras.io → keras.io; the TensorFlow for Poets codelab is now deprecated as of TF 2.0), so today's live page differs from the 2018 version even though the resource existed then.
- **The saree taxonomy is the fabricated core of the blueprint.** No verified 2018 resource defines the eight named saree pattern categories or a "saree design fingerprint"; those are original domain logic layered on top of generic tools.
- **The handloom citation caution is important:** at least one commonly-circulated "Indian handloom fabric classification 2018" citation is garbled/misattributed (its method belongs to Huang & Fu's fleece-pilling paper). I have flagged it rather than repeating it as fact.
- I was unable to run final confirmation searches on the exact 2018 SQLite/node-sqlite3 tutorial pages and the OpenCV histogram/colorspaces page due to a search-budget limit; those layers are supported by the libraries' canonical documentation, which unquestionably existed in 2018, but I did not capture a specific dated tutorial URL for the node-sqlite3 binding. All other layers are backed by dated, named sources.