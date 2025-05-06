# Multimodal task: Image Text Analysis and Generation system

Based on the multimodal analysis method, 
this project has constructed a complete image-text 
processing and generation system around the 
relationship between images and texts. 
The project structure is clear and the functions 
are divided into four modules: 'crawl', 
'api_annotation', 'visualization' and 'data_mine', 
which are respectively used for data collection, 
annotation, visualization and deep learning modeling tasks.
---

## 📁 Module structure description

### 1. 'crawl' : Image data crawling module
It is used to crawl image resources from the network and save them as file structures by class for subsequent use.
Implement web page automated crawlers using 'Selenium' + 'BeautifulSoup'
Multi-threaded image download and support retry mechanism
The downloaded images can be used for classification tasks or image-text matching tasks

### 2. 'api_annotation' : Image automatic annotation module
Automatically label the images using the external API and return the text description information.
The data is divided into six dimensions and a Hellenized descriptive statement to associate the house pictures with emotions
Relate them together to establish a supervised data set.
Image path reading and API request processing
Support batch image annotation and result saving
Output in CSV file format, including image paths, text descriptions, and multi-dimensional scoring information

Sample output field:
```csv
filename,Safety,Belonging,Naturalness,Shared,Privacy,Convenience,concept_text
example.jpg,3,4,2,5,3,4,"a modern open space"
```

### 3. 'visualization' : Visualization module
Visualize the relationship between conceptual texts and multi-dimensional scoring to assist in understanding the quality and patterns of annotations.
Use 'matplotlib' and 'seaborn' to draw the distribution map of scoring dimensions
Supports various display forms such as' boxplot 'and' scatter '
- the input file: ` api_annotation/annotations. CSV `

### 4. 'data_mine' : Data Mining and Image Generation Module
It includes contents such as text encoding, GAN model training and evaluation, etc. The core task is "text-to-image generation".
Obtain the text vector based on 'CLIP'
Conditional GAN has been implemented, integrating text semantics for image generation
Supports reasoning interfaces for visualizing the training process, saving models, and generating images from text

Sample training output:
```
output/sample_epoch_1.png
checkpoints/generator_epoch.pth
```

---

## 📦 Depends on the environment

Please use the Python 3.8+ environment and make sure to install the following dependencies:
```bash
pip install torch torchvision tqdm pandas matplotlib seaborn clip-by-openai
```

---

## 🚀 Start quickly

1. **Crawl images**
   ```bash
   cd crawl
   python run_crawler.py
   ```

2. **API Label image**
   ```bash
   cd api_annotation
   python annotate.py
   ```

3. **Visual analysis**
   ```bash
   cd visualization
   python visualize.py
   ```

4. **Train and generate images**
   ```bash
   cd data_mine
   python GAN.py      # Train the Model
   python eval.py     # Text-to-image generation
   ```

---

## 📄 File structure
```
project_root/
│
├── crawl/                # Image crawling module
├── api_annotation/       # Image annotation module
├── visualization/        # Analysis and visualization
├── data_mine/            # Data mining and image generation
│   ├── GAN.py            # GAN training script
│   ├── eval.py           # Text-to-image generation script
│   └── checkpoints/      # Model save path
│
└── README.md             # This file

```

---
