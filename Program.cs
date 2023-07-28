﻿using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;


class CLIP {
    static void Main(string[] args) {
        // Load the model
        // Model sourced from: https://huggingface.co/rocca/openai-clip-js/tree/main
        var clipModel = new InferenceSession("../../../clip-image-vit-32-float32.onnx");
        
        // Load a sample image
        var image = Image.Load<Rgba32>(File.ReadAllBytes("../../../astronaut.png"));
        
        // Resize to 224 x 224 (bicubic resizing is the default)
        image.Mutate(x => x.Resize(224, 224));

        // Create a new array for 1 picture, 3 channels (RGB) and 224 pixels height and width
        var inputTensor = new DenseTensor<float>(new[] {1, 3, 224, 224});
        
        // Put all the pixels in the input tensor
        for (var x = 0; x < 224; x++)
        {
            for (var y = 0; y < 224; y++)
            {
                // Normalize from bytes (0-255) to floats (constants borrowed from CLIP repository)
                inputTensor[0, 0, y, x] = Convert.ToSingle((((float) image[x, y].R / 255) - 0.48145466) / 0.26862954);
                inputTensor[0, 1, y, x] = Convert.ToSingle((((float) image[x, y].G / 255) - 0.4578275 ) / 0.26130258);
                inputTensor[0, 2, y, x] = Convert.ToSingle((((float) image[x, y].B / 255) - 0.40821073) / 0.27577711);
            }
        }

        // Prepare the inputs as a named ONNX variable, name should be "input"
        var inputs = new List<NamedOnnxValue> {NamedOnnxValue.CreateFromTensor("input", inputTensor)};
        
        // Run the model, and get the output back as an Array of floats
        var outputData = clipModel.Run(inputs).ToList().Last().AsTensor<float>().ToArray();
        
    }
}


