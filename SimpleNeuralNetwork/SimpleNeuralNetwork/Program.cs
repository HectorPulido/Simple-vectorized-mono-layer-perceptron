using System;
using LinearAlgebra;

/*
 THIS PROGRAM USES SIMPLE LINEAR ALGEBRA LIBRARY FROM GITHUB
 https://github.com/HectorPulido/Simple_Linear_Algebra     
*/
namespace SimpleNeuralNetwork
{
    class Program
    {
        const double Epsilon = 0.01; // Epsilon is a value near to 0

        static void Main(string[] args)
        {
            //Parameters
            int inputCount = 2;
            int hiddenCount = 5;
            int outputCount = 4;
            int examplesCount = 4;
            double learningRate = 0.1;
            Random r = new Random();

            //Training data           //INPUT
            Matrix x = new double[,] { { 0, 0 }, 
                                       { 0, 1 }, 
                                       { 1, 0 }, 
                                       { 1, 1 }};  
                                    //DESIRED OUTPUT
                                    //XNOR AND OR XOR
            Matrix y = new double[,] { { 1, 0, 0, 0 }, 
                                       { 0, 0, 1, 1 }, 
                                       { 0, 1, 1, 1 }, 
                                       { 1, 1, 1, 0 }}; 

            //Weight init
            Matrix w1 = (Matrix.Random(inputCount + 1, hiddenCount , r) - 0.5) * 2.0;
            Matrix w2 = (Matrix.Random(hiddenCount + 1, outputCount, r) - 0.5) * 2.0;

            // TO USE RELU
            // 0. Change all sigmoid function, for relu function
            // 1. a3 must have no Nonlinear function Matrix a3 = z3;
            // 2. because of that Delta3 has not derivated Matrix Delta3 = a3Error * 1;
            // 3. The learning rate must be smaller, like 0.001            

            for (int l = 0; l < 5001; l++) //epoch
            {
                //Forward pass
                Matrix z1 = x.AddColumn(Matrix.Ones(examplesCount, 1));
                Matrix a1 = z1; //(Examples, input + 1)
                Matrix z2 = (a1 * w1).AddColumn(Matrix.Ones(examplesCount, 1));
                Matrix a2 = sigmoid(z2); //(examples, hidden neurons + 1) // APPLY NON LINEAR
                Matrix z3 = a2 * w2;
                Matrix a3 = sigmoid(z3); //(examples, output)                  

                //Bacpropagation
                Matrix a3Error = a3 - y; //(examples, output) //LOSS 
                Matrix Delta3 = a3Error * sigmoid(z3, true);

                Matrix a2Error = Delta3 * w2.T;
                Matrix Delta2 = a2Error * sigmoid(z2, true);
                Delta2 = Delta2.Slice(0, 1, Delta2.x, Delta2.y); //Slicing Extra delta (from biass neuron)

                w2 -= (a2.T * Delta3) * learningRate;
                w1 -= (a1.T * Delta2) * learningRate;

                double loss = a3Error.abs.average * examplesCount;
                Console.WriteLine("Loss: " + loss);

                if (l % 1000 == 0)
                {
                    Console.WriteLine("---------"+l+"----------------------------------------------------");
                    Console.WriteLine("X: " + x.size.ToString());
                    Console.WriteLine(x.ToString());

                    Console.WriteLine("Prediction: " + a3.size.ToString());
                    Console.WriteLine(a3);                                                
                }
            }
            Console.ReadKey();            
        }

        static Matrix sigmoid(Matrix m, bool derivated = false)
        {
            double[,] a = m;
            Matrix.MatrixLoop((i, j) =>
            {
                if (derivated)
                {
                    double sig = 1.0 / (1.0 + Math.Exp(-a[i, j]));
                    a[i, j] = sig * (1.0 - sig);
                }
                else
                {
                    a[i, j] = 1.0 / (1.0 + Math.Exp(-a[i, j]));
                }
            }, m.x, m.y);

            return a;
        }
        static Matrix relu(Matrix m, bool derivated = false)
        {
            double[,] a = m;
            Matrix.MatrixLoop((i, j) =>
            {
                if (derivated)
                {
                    a[i, j] = a[i,j] > 0 ? 1 : Epsilon;
                }
                else
                {
                    a[i, j] = a[i, j] > 0 ? a[i, j] : 0;
                }
            }, m.x, m.y);

            return a;
        }
    }
}
