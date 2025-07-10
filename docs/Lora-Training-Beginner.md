# LoRA Training Guide for Stable Diffusion 3.5 Large - Complete Beginner

## What is LoRA Training?

LoRA (Low-Rank Adaptation) is a way to teach an AI image generator to create pictures in a specific style or of a specific person/object. Think of it like showing the AI many examples of what you want, so it can learn to make similar images when you ask for them.

**Stable Diffusion 3.5 Large** is a powerful AI that can create images from text descriptions. By training a LoRA, you're customizing this AI to understand your specific needs.

## What You'll Need

* A computer with internet access
* 10-50 pictures of what you want the AI to learn (the more similar, the better)
* About 1-2 hours of your time
* Access to the LoRA training website/platform

## Step-by-Step Training Process

### Step 1: Gather Your Images

**What to do:** Collect pictures of what you want the AI to learn.

**Detailed explanation:**

* Choose 15-30 high-quality photos (more isn't always better!)
* All pictures should show the same person, style, or object you want to teach the AI
* Make sure pictures are clear and not blurry
* Bigger pictures work better (at least 512 pixels wide/tall)
* Remove any photos that are very dark, blurry, or poor quality

**Example:** If you want to train the AI to draw your pet dog, collect 20-30 clear photos of your dog from different angles.

### Step 2: Add Captions for Each Picture (Optional)

**What to do:** Write a short description for each picture, or skip this step.

**Detailed explanation:**
A "caption" is just a sentence describing what's in the picture. For example: "a golden retriever sitting in grass" or "a woman with brown hair wearing a red dress."

* You can write these descriptions yourself for better control
* Or you can skip this step and let the computer write them automatically (Step 4)
* If you write them yourself, use similar words across all descriptions
* Keep descriptions simple and factual

**Beginner tip:** If this sounds complicated, just skip it! The automatic option works great.

### Step 3: Put Your Images in a Zip File and Upload

**What to do:** Package all your pictures into one file and upload it to the training website.

**Detailed explanation:**
A ".zip file" is like a digital folder that squishes multiple files together into one package.

**How to create a zip file:**

* On Windows: Select all your images, right-click, choose "Send to" → "Compressed (zipped) folder"
* On Mac: Select all your images, right-click, choose "Compress items"
* On phones: Use a file manager app that can create zip files

**Important:** Don't put your images in folders inside the zip file - just put all the pictures directly in the zip file.

Then upload this zip file to the training platform through their website.

### Step 4: Choose Auto Caption (If You Skipped Step 2)

**What to do:** Tell the computer to automatically write descriptions for your pictures.

**Detailed explanation:**
If you didn't write captions in Step 2, the computer can look at your pictures and automatically write descriptions. This is pretty smart and works well most of the time.

* Just check the "auto caption" option on the website
* The computer will analyze what's in each picture
* It will write descriptions like "a dog sitting on grass" or "a person wearing glasses"

### Step 5: Choose Your Trigger Word

**What to do:** Pick a special word that will make the AI use your trained style.

**Detailed explanation:**
A "trigger word" is like a magic word you'll use in your image prompts to activate your custom training. When you type this word, the AI will know to use what it learned from your pictures.

**Good trigger words:**

* Made-up names like "Z3ph" or "Bl1xt0r" using numbers for vowels
* Unlikely character combinations like "XJX" or "KZK"

**Avoid:** Common words like "dog," "person," or "style" (these might confuse the AI)

**Example:** If your trigger word is "Bl1xt0r" you'll later create images by typing prompts like "Bl1xt0r sitting in a garden" or "Bl1xt0r wearing a blue dress."

### Step 6: Set the Optimizer to Prodigy with Learning Rate 1.0

**What to do:** Choose specific technical settings (don't worry, this is easier than it sounds!)

**Detailed explanation:**
These are technical settings that control how the AI learns. Think of them like the "speed" and "method" of learning.

* **Optimizer = Prodigy:** This is the learning method. Prodigy is smart and adjusts itself automatically.
* **Learning Rate = 1.0:** This is the learning speed. With Prodigy, 1.0 is the perfect speed.

**Beginner tip:** Just set these exactly as written - they're the best settings for beginners and experts alike!

### Step 7: Increase Batch Size to 4

**What to do:** Change the "batch size" setting from whatever it is to 4.

**Detailed explanation:**
"Batch size" means how many pictures the AI looks at at the same time while learning. Think of it like studying - some people learn better by looking at one flashcard at a time, others learn better by looking at several flashcards together.

* Setting this to 4 makes training faster and more stable
* It's like the AI studying 4 pictures at once instead of 1
* This helps the AI learn patterns better

### Step 8: Set Training Steps to 1000-2000

**What to do:** Tell the AI how many times to practice learning from your pictures.

**Detailed explanation:**
"Steps" are like practice rounds. Each step, the AI looks at your pictures and gets a little bit better at understanding them.

* **1000 steps:** Good for smaller collections (10-20 pictures)
* **1500 steps:** Good for medium collections (20-80 pictures)
* **2000 steps:** Good for larger collections (80+ pictures)

**Think of it like:** If you were learning to draw, 1000 steps would be like practicing for 1000 sessions.

### Step 9: Set Resolution to "768,1024"

**What to do:** Type "768,1024" in the resolution setting.

**Detailed explanation:**
This setting helps the AI learn from pictures of different shapes - both tall pictures and wide pictures.

* The AI uses something called "aspect bucketing" (fancy term, but you don't need to understand it)
* This just means the AI can learn from both portrait photos (tall) and landscape photos (wide)
* "768,1024" tells the AI the maximum sizes to work with
* This is much better than forcing all pictures to be the same shape

**Beginner tip:** Just type exactly "768,1024" - the comma is important!

### Step 10: Start Training!

**What to do:** Click the "Start Training" button and wait.

**Detailed explanation:**
Now the AI will spend time learning from your pictures. This usually takes 30 minutes to 2 hours depending on how many pictures you uploaded and how busy the training servers are.

You can:

* Close your browser and come back later
* Check on progress if the website shows a progress bar
* Be patient - good training takes time!

## After Training is Complete

### Getting Your LoRA File

**What to do:** Download your trained LoRA and rename it.

**Detailed explanation:**
When training finishes, you'll get a ".tar" file download. This is like a zip file containing your trained LoRA.

**Steps:**

1. Download the .tar file
2. Open/extract it (your computer should be able to do this automatically)
3. Find the ".safetensors" file inside
4. Rename it to something memorable like "my\_dog\_lora.safetensors" or "sarah\_portrait\_style.safetensors"

### Using Your LoRA

**What to do:** Load your LoRA into an AI image generator and start creating!

**Detailed explanation:**
To use your trained LoRA:

1. Open your favorite AI image generator that supports Stable Diffusion 3.5 Large
2. Load/import your .safetensors file
3. In your text prompts, include your trigger word
4. Set LoRA strength between 0.7 and 1.0 (start with 0.8)

**Example prompts with trigger word "**Bl1xt0r**":**

* "Bl1xt0r sitting in a beautiful garden"
* "Bl1xt0r wearing elegant evening dress"
* "portrait of Bl1xt0r with soft lighting"

## Beginner Tips for Success

### Image Quality Matters Most

* 20 great pictures are better than 50 okay pictures
* Make sure photos are clear and well-lit
* Remove any blurry or very dark photos

### Keep It Consistent

* If training on a person, use photos where they look similar
* If training on a style, make sure the style is consistent across all images
* Avoid mixing completely different styles or subjects

### Don't Overthink It

* The default settings work great for most people
* Start simple - you can retrain again with iterative changes to different settings
* It's okay if you don't understand all the technical terms

### Common Beginner Mistakes to Avoid

* Using too many different types of images in one training
* Choosing common words as trigger words
* Expecting perfect results immediately (it takes practice!)
* Using low-quality or very small images

## Glossary of Terms

**LoRA:** Low-Rank Adaptation - a way to customize AI models
**Stable Diffusion:** The AI that creates images from text descriptions  
**Trigger Word:** Special word that activates your custom training
**Batch Size:** How many images the AI studies at once
**Steps:** How many practice rounds the AI goes through
**Resolution:** Image size settings
**Optimizer:** The method the AI uses to learn
**Learning Rate:** How fast the AI learns
**Safetensors:** The file format that contains your trained LoRA
**.tar file:** A compressed file containing your finished LoRA
**Aspect Bucketing:** Technical feature that lets AI learn from different image shapes

Remember: Everyone was a beginner once! Don't be afraid to experiment and try again if your first attempt isn't perfect.

