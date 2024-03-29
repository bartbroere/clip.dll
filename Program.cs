using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;


class CLIP {
    static void Main(string[] args) {
        // Download the model weights if we don't have them in the current directory
        if (!File.Exists("clip-image-vit-32-float32.onnx"))
        {
            WebClient webClient = new WebClient();
            webClient.DownloadFile(
                "https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-image-vit-32-float32.onnx",
                @"clip-image-vit-32-float32.onnx"
            );
        }

        // Load the model
        // Model sourced from: https://huggingface.co/rocca/openai-clip-js/tree/main
        var clipModel = new InferenceSession("clip-image-vit-32-float32.onnx");

        // Load an image specified as a command line argument
        var image = Image.Load<Rgba32>(File.ReadAllBytes(args[0]));

        // Calculate the shortest side, and use that to extract a square from the center
        // Known in other image libraries as Centercrop
        // AFAIK Centercrop is not available in Sixlabors.ImageSharp, so we do it manually
        var smallestSide = Math.Min(image.Width, image.Height);
        image.Mutate(x => x.Crop(
            new Rectangle(
                (image.Width - smallestSide) / 2,
            (image.Height - smallestSide) / 2,
            smallestSide,
            smallestSide
        )));

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

        // Write the array serialized as JSON
        Console.WriteLine(JsonSerializer.Serialize(outputData));
        
    }
}


