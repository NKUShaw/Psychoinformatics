# Post-hoc and manifold explanations analysis of facial expression-psychological traits data based on deep learning

#### Authors: 
- [Yang Xiao](https://scholar.google.co.uk/citations?hl=zh-TW&user=FvnT29sAAAAJ)(2013241@mail.nankai.edu.cn) (yax3417@utulsa.edu)

#### Graduate Thesis: 
 - [Chinese]**基于深度学习的面部表情识别研究**
 - [English]**[Post-hoc and manifold explanations analysis of facial expression-psychological traits data based on deep learning](http://arxiv.org/abs/2404.18352)**

#### Abstract: 
The complex information processing system of humans generates a lot of objective and subjective evaluations, making the exploration of human cognitive products of great cutting-edge theoretical value. In recent years, deep learning technologies, which are inspired by biological brain mechanisms, have made significant strides in the application of psychological or cognitive scientific research, particularly in the memorization and recognition of facial data. This paper investigates through experimental research how neural networks process and store facial expression data and associate these data with a range of psychological attributes produced by humans. Researchers utilized deep learning models such as CapsNet, VGG16, and Transformer, demonstrating that neural networks can learn and reproduce key features of facial data, thereby storing image memories. Moreover, the experimental results reveal the potential of deep learning models in understanding human emotions and cognitive processes and establish a manifold visualization interpretation of cognitive products or psychological attributes from a non-Euclidean space perspective, offering new insights into enhancing the explainability of AI. This study not only advances the application of AI technology in the field of psychology but also provides a new theoretical basis for understanding the information processing of the human brain.

#### Key Words: Psychological attributes; Neural network; Machine learning; Deep learning; Manifold learning

#### Notice: 
Due to storage limitations, the weight files cannot be uploaded. If you need to make a comparison by testing in my weight files, please contact me via email.

#### Please run the following command to do train process.
```
python {Memory/Psychology}/main.py --model={VGG16/VIT/Others} --batch_size=40 --lr=5e-4 --target={whatyoupredict} 
```


