# Data Collection and Annotation Guide

## Object Classes

0. mouse
1. keyboard
2. laptop
3. cup
4. headphones

## Collection Sessions

Each contributor performs the following collection tasks.

### S01 - Basic

Normal lighting and relatively simple background.

- single-object images
- different viewing angles
- normal viewing distance
- approximately 50 images

### S02 - Multi-object

Images containing 2-5 target objects.

- different object combinations
- different positions
- approximately 50 images

### S03 - Environment

Change environmental conditions.

- different tables
- different rooms
- different backgrounds
- different lighting
- approximately 40 images

### S04 - Hard Samples

More difficult situations.

- partial occlusion
- edge objects
- longer viewing distance
- cluttered backgrounds
- overlapping objects
- approximately 25 images

### S05 - Negative Samples

Images containing none of the five target classes.

Examples:

- books
- phones
- chargers
- paper
- bags

Approximately 15 images.

## Image Naming

    PXX_SXX_TYPE_XXXX.jpg

Object type codes:

    MOU = mouse
    KEY = keyboard
    LAP = laptop
    CUP = cup
    HDP = headphones
    MIX = multiple target classes
    NEG = negative sample

Example:

    P03_S02_MIX_0017.jpg

## Annotation Rules

Each visible object instance receives one bounding box.

- mouse: annotate the mouse body; long cables are excluded
- keyboard: annotate only independent external keyboards
- laptop: annotate the entire visible laptop
- cup: include cup body and handle
- headphones: annotate the headphone body; long cables are excluded
- laptop built-in keyboards are NOT separately labeled as keyboard
- bounding boxes should fit object boundaries closely
- separate objects of the same class must receive separate boxes
- seriously occluded objects should normally be excluded from the first dataset version

## YOLO Label Format

Each image corresponds to a TXT label file with the same base filename.

Example:

    P03_S02_MIX_0017.jpg
    P03_S02_MIX_0017.txt

YOLO label format:

    class_id x_center y_center width height

All coordinate values are normalized to the range 0-1.
