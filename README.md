### 1. How to run the code?
   #### Connect to GPU and run the sections in order:
   	Install required libraries
	
	Import dataset: CIFAR10 or Beans dataset
	
	Either run cells from 'ViT Model' or 'Learnable resizing network + ViT' sections
 	
	Download the checkpoints (if required)
	
   #### Note:  
   1. Current version of notebook runs for Beans dataset. For CIFAR10 dataset, uncomment and comment codes as indicated in the notebook.  
   2. When prompted login using weights and biases account to store all results/graphs.   
   3. If not required, comment out **report_to=wandb** parameter in the args variable for the Trainer (hugging face Trainer) (Comment code as indicated in the notebook)
   #### Required additional libraries
	+ pip install transformers
	+ pip install datasets
	+ pip install wandb

### 2. Codes used

  #### (a) Repositories used 
	
1. [Fine tuning Vision Transformers using the hugging face 🤗 trainer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_🤗_Trainer.ipynb)
	
2. [Learning to resize images for computer vision tasks](https://github.com/yundaehyuck/Learning-to-resize-images-for-computer-vision-tasks/blob/main/resizing_network.ipynb)
  
  #### (b) Modified code

	 Modified the torchvision.transforms (train_transforms/val_transforms function) in [1] for
			+ Adjusting sizes for Resizer+Vit Model
			+ Making it work for Beans dataset
	 Modified args = TrainingArguments(...) to add
			+ report_to="wandb"
			+ logging_steps = len(train_ds)//64

  #### (c) My original code
	 
 	Running the models on Beans dataset (Modifying parts of code accordingly)
	
	Built a custom model combining Resizer network [2] and ViT Model [1] (Resizer_ViT Class).
	
	Added the wandb (Weights and Biases to visualize the output: Training and testing losses and evaluation accuracy)
	
	Visualize learnable resizer output
	 	- Wrote code to pull out the output of the resizer network to see what images are being fed into the ViT model in the Resizer_ViT model

	Sections of Original Code:
		- Custom Model: Resizer_ViT model
		- Predictions on the test dataset: ViT
		- Predictions on the test dataset: Resizer_ViT
		- Visualize results
		- Visualize learnable resizer output
### 3. Datasets used

	a) CIFAR10 dataset (1k images)
		Original dataset: 50k images for training, 10k for testing, 10 classes
		Modified dataset used: 1k images for training (900 for training and 100 for validation), 1k images for testing
	
	b) Beans disease classification dataset
		Dataset description:
			- Total images: 1034 images each of 500x500 resolution for training (train_val split=0.1) 
			- and 128 images for testing

	Both datasets were available in hugging face package.
			- See Import datasets section of attached ipynb notebook
### 4. Presentation video

[Here](https://youtu.be/WLj-Kjk-TYk) is the link to my presentation of this project.

### 5. References

[1] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020). [[paper]](https://arxiv.org/pdf/2010.11929.pdf)

[2] Talebi, Hossein, and Peyman Milanfar. "Learning to resize images for computer vision tasks." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [[paper]](https://arxiv.org/pdf/2103.09950.pdf)
	
**Varun Machhale Kumar  
Purdue University**
