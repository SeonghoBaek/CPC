# CPC
CPC based Representation Learning 


Input: [96 x 96 x 3]
Patch: [24 x 24 x 3], Overlap size 12. 1 input image is composed with 49 patches.

Flow: Input Image -> Make Patches -> Encoding -> Pixel CNN -> CPC

Encoder: ResDense Block + Global AVG Pool, No Pooling Layer(Conv Only), Batch Norm and Self Attention.
