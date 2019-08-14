using System;
using static System.Console;
using System.Diagnostics;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;

#pragma warning disable CS0649

namespace myApp
{
    class Program
    {
        // STEP 1: Define your data structures
        // IrisData is used to provide training data, and as
        // input for prediction operations
        // - First 4 properties are inputs/features used to predict the label
        // - Label is what you are predicting, and is only set when training
        public class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        // IrisPrediction is the result returned from prediction operations
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        static void Main(string[] args)
        {
            // STEP 2: Create a ML.NET environment  
            var mlContext = new MLContext();
            Random rnd = new Random();
            Stopwatch sw = new Stopwatch();

            Write("Loading dataset... ");
            sw.Start();
            // If working in Visual Studio, make sure the 'Copy to Output Directory'
            // property of iris-data.txt is set to 'Copy always'
            var reader = mlContext.Data.CreateTextLoader<IrisData>(separatorChar: ',', hasHeader: false);
            
            //https://en.m.wikipedia.org/wiki/Iris_flower_data_set
            IDataView trainingDataView = reader.Load("iris-data.txt");

            // STEP 3: Transform your data and add a learner
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training.
            // Add a learning algorithm to the pipeline. e.g.(What type of iris is this?)
            // Convert the Label back into original text (after converting to number in step 3)
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            sw.Stop();
            WriteLine($"{sw.Elapsed.Seconds*1000 + sw.Elapsed.Milliseconds} ms");

            // STEP 4: Train your model based on the data set  
            Write("Training the model... ");
            sw.Restart();
            var model = pipeline.Fit(trainingDataView);
            sw.Stop();
            WriteLine($"{sw.Elapsed.Seconds*1000 + sw.Elapsed.Milliseconds} ms\n");
            
            do {
                // STEP 5: Use your model to make a prediction
                // You can change these numbers to test different predictions
                var iris = new IrisData()
                    {
    //                    SepalLength = 2.3f,
    //                    SepalWidth = 1.6f,
    //                    PetalLength = 0.2f,
    //                    PetalWidth = 5.1f,
                        SepalLength = (float)rnd.NextDouble() * 3.6f + 4.3f,
                        SepalWidth = (float)rnd.NextDouble() * 2.4f + 2.0f,
                        PetalLength = (float)rnd.NextDouble() * 5.9f + 1.0f,
                        PetalWidth = (float)rnd.NextDouble() * 2.4f,
                    };
    
                WriteLine($"Sepal length: {iris.SepalLength:F1} cm");
                WriteLine($"Sepal width:  {iris.SepalWidth:F1} cm");
                WriteLine($"Petal length: {iris.PetalLength:F1} cm");
                WriteLine($"Petal width:  {iris.PetalWidth:F1} cm");
    
                var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(iris);
                WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
                WriteLine("Esc to quit...\n");
                
            } while (ReadKey(true).Key != ConsoleKey.Escape);

        }
    }
}
#pragma warning restore CS0649