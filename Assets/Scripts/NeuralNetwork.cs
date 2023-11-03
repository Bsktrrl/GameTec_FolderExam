using System.Collections.Generic;
using UnityEngine;

public enum TrainingModel
{
    Sigmoid,
    Relu
}

public class NeuralNetwork 
{
    #region variables
    //Nodes in each layer of the network
    int inputNodes_Amount = 0;
    List<int> hiddenNodesList = new List<int>();
    int outputNodes_Amount = 0;

    float learningRate = 0.1f;

    TrainingModel trainingModel = TrainingModel.Sigmoid;

    //Weighs to change the numbers going to different parts of the network
    Matrix hiddenWeights_Input;
    List<Matrix> hiddenWeightList = new List<Matrix>();
    Matrix hiddenWeights_Output;

    //Layers of Neurons
    Matrix layer_Imput = new Matrix();
    List<Matrix> hiddenLayerList = new List<Matrix>();
    Matrix layer_Output = new Matrix();

    //Bias
    List<Matrix> hiddenBiasList = new List<Matrix>();
    Matrix bias_Output;
    #endregion


    //--------------------


    public void StartUp(int _inputNodes_Amount, List<int> _hiddenNodesList, int _outputNodes_Amount)
    {
        hiddenNodesList = _hiddenNodesList;
        inputNodes_Amount = _inputNodes_Amount;
        outputNodes_Amount = _outputNodes_Amount;

        //Set Bias Output
        hiddenBiasList = new List<Matrix>();

        for (int i = 0; i < _hiddenNodesList.Count; i++)
        {
            hiddenBiasList.Add(new Matrix(_hiddenNodesList[i], 1));
            hiddenBiasList[i].Randomize();
        }

        bias_Output = new Matrix(_outputNodes_Amount, 1);
        bias_Output.Randomize();

        //Set Output Weights
        hiddenWeights_Input = new Matrix(_hiddenNodesList[0], _inputNodes_Amount);
        hiddenWeights_Input.Randomize();
        hiddenWeightList = new List<Matrix>();

        for (int i = 1; i < _hiddenNodesList.Count; i++)
        {
            hiddenWeightList.Add(new Matrix(_hiddenNodesList[i], _hiddenNodesList[i - 1]));
            hiddenWeightList[i - 1].Randomize();
        }

        hiddenWeights_Output = new Matrix(_outputNodes_Amount, _hiddenNodesList[_hiddenNodesList.Count - 1]);
        hiddenWeights_Output.Randomize();


        //Let Output Layers
        layer_Imput = new Matrix(_inputNodes_Amount, 1);
        hiddenLayerList = new List<Matrix>();

        for (int i = 0; i < _hiddenNodesList.Count; i++)
        { 
            hiddenLayerList.Add(new Matrix(_hiddenNodesList[i], 1));
        }

        layer_Output = new Matrix(_outputNodes_Amount,1);
    }
    public List<float> FeedForward(List<float> inputs)
    {
        //Move data one step further into the network

        float[] inputList = inputs.ToArray();
        Matrix matrix_Input = new Matrix(inputNodes_Amount, 1);

        for (int i = 0; i < matrix_Input.rows; i++)
        {
            matrix_Input.mat[i][0] = inputList[i];
        }

        layer_Imput = matrix_Input;

        //Setup the first element in the "hiddenLayerList"
        hiddenLayerList[0] = new Matrix(Matrix.MatrixProduct(hiddenWeights_Input.mat, matrix_Input.mat));
        hiddenLayerList[0] = hiddenLayerList[0].Add(hiddenBiasList[0]);

        switch(trainingModel)
        {
            case TrainingModel.Sigmoid:
                hiddenLayerList[0] = hiddenLayerList[0].Sigmoid();
                break;
            case TrainingModel.Relu:
                hiddenLayerList[0] = hiddenLayerList[0].Relu();
                break;

            default:
                break;
        }

        //Go through the rest of the "hiddenLayerList" 
        for (int i = 1; i < hiddenLayerList.Count; i++)
        {
            hiddenLayerList[i] = new Matrix(Matrix.MatrixProduct(hiddenWeightList[i - 1].mat, hiddenLayerList[i - 1].mat));
            hiddenLayerList[i] = hiddenLayerList[i].Add(hiddenBiasList[i]);

            switch (trainingModel)
            {
                case TrainingModel.Sigmoid:
                    hiddenLayerList[i] = hiddenLayerList[i].Sigmoid();
                    break;
                case TrainingModel.Relu:
                    hiddenLayerList[i] = hiddenLayerList[i].Relu();
                    break;

                default:
                    break;
            }
        }

        //Get the layer_Output from the hidden layer
        layer_Output = new Matrix(Matrix.MatrixProduct(hiddenWeights_Output.mat, hiddenLayerList[hiddenLayerList.Count - 1].mat));
        layer_Output = layer_Output.Add(bias_Output);

        switch (trainingModel)
        {
            case TrainingModel.Sigmoid:
                layer_Output = layer_Output.Sigmoid();
                break;
            case TrainingModel.Relu:
                layer_Output = layer_Output.Relu();
                break;
            default:
                break;
        }

        //transfer the Matrix over to a List<float> of outputs
        List<float> outputList = new List<float>();
        for (int i = 0; i < layer_Output.rows; i++)
        {
            outputList.Add(layer_Output.mat[i][0]);
        }

        return outputList;
    }
    public void BackPropagate(List<float> output, List<float> direction_outputs)
    {
        //Perform supervised learning by going backwards inthe network, feeding ned data based on the output

        //Make a new Matrix, feeding in the outputList of the network
        float[] outputList = output.ToArray();
        Matrix matrix_Output = new Matrix(output.Count, 1);

        for (int i = 0; i < matrix_Output.rows; i++)
        {
            matrix_Output.mat[i][0] = outputList[i];
        }

        #region Errors
        //Find the error of the output
        Matrix error_Output = new Matrix(output.Count, 1);
        for (int i = 0; i < direction_outputs.Count && i < output.Count; i++)
        {
            error_Output.mat[i][0] = direction_outputs[i] - output[i];
        }

        //Find the hidden weights
        Matrix hiddenOutpuWeight_Transposed = hiddenWeights_Output.Transpose();
        List<Matrix> hiddenOutputWeight_TransposedList = new List<Matrix>();
        List<Matrix> hiddenWeight_Transposed = new List<Matrix>();

        for (int i = hiddenWeightList.Count - 1; i >= 0; i--)
        {
            hiddenOutputWeight_TransposedList.Add(hiddenWeightList[i].Transpose());
        }
        for (int i = 0; i < hiddenWeightList.Count; i++)
        {
            hiddenWeight_Transposed.Add(hiddenOutputWeight_TransposedList[i]);
        }

        //Find the hidden errors
        List<Matrix> hiddenErrorList = new List<Matrix>();
        List<Matrix> hiddenError_TempList = new List<Matrix>();
        hiddenError_TempList.Add(new Matrix(Matrix.MatrixProduct(hiddenOutpuWeight_Transposed.mat, error_Output.mat)));

        for (int i = 0; i < hiddenWeight_Transposed.Count; i++)
        {
            hiddenError_TempList.Add(new Matrix(Matrix.MatrixProduct(hiddenWeight_Transposed[i].mat, hiddenError_TempList[i].mat)));
        }
        for (int i = hiddenError_TempList.Count - 1; i >= 0; i--)
        {
            hiddenErrorList.Add(hiddenError_TempList[i]);
        }
        #endregion

        #region Output layer
        //Find the directen to move the correction in, and the amount to correct with
        Matrix output_gradient = new Matrix();
        switch (trainingModel)
        {
            case TrainingModel.Sigmoid:
                output_gradient = matrix_Output.SigmoidPrime();
                break;
            case TrainingModel.Relu:
                output_gradient = matrix_Output.ReluPrime();
                break;
            default:
                break;
        }

        output_gradient = output_gradient.Multiply(error_Output);
        output_gradient = output_gradient.Multiply(learningRate);

        //Add the correction
        Matrix hidden_T = hiddenLayerList[hiddenLayerList.Count - 1].Transpose();
        Matrix hidden_output_weight_delta = new Matrix(Matrix.MatrixProduct(output_gradient.mat, hidden_T.mat));

        hiddenWeights_Output = hiddenWeights_Output.Add(hidden_output_weight_delta);
        bias_Output = bias_Output.Add(output_gradient);
        #endregion

        #region Hidden layers
        //Find the hidden gradient and add it to the correction
        for (int i = hiddenLayerList.Count - 1; i >= 1; i--)
        {
            Matrix hiddenGradient = new Matrix();

            switch (trainingModel)
            {
                case TrainingModel.Sigmoid:
                    hiddenGradient = hiddenLayerList[i].SigmoidPrime();
                    break;
                case TrainingModel.Relu:
                    hiddenGradient = hiddenLayerList[i].ReluPrime();
                    break;

                default:
                    break;
            }

            hiddenGradient = hiddenGradient.Multiply(hiddenErrorList[i]);
            hiddenGradient = hiddenGradient.Multiply(learningRate);

            Matrix hiddenTransposed = hiddenLayerList[i - 1].Transpose();
            Matrix hiddenWeights_Delta = new Matrix(Matrix.MatrixProduct(hiddenGradient.mat, hiddenTransposed.mat));

            hiddenWeightList[i - 1] = hiddenWeightList[i - 1].Add(hiddenWeights_Delta);
            hiddenBiasList[i] = hiddenBiasList[i].Add(hiddenGradient);
        }

        //Find the directen to move the correction in, and the amount to correct with
        Matrix gradient_Hidden = new Matrix();
        switch (trainingModel)
        {
            case TrainingModel.Sigmoid:
                gradient_Hidden = hiddenLayerList[0].SigmoidPrime();
                break;
            case TrainingModel.Relu:
                gradient_Hidden = hiddenLayerList[0].ReluPrime();
                break;
            default:
                break;
        }

        gradient_Hidden = gradient_Hidden.Multiply(hiddenErrorList[0]);
        gradient_Hidden = gradient_Hidden.Multiply(learningRate);

        //Add the correction
        Matrix inputTransposed = layer_Imput.Transpose();
        Matrix hiddenWeight_InputDelta = new Matrix(Matrix.MatrixProduct(gradient_Hidden.mat, inputTransposed.mat));

        hiddenWeights_Input = hiddenWeights_Input.Add(hiddenWeight_InputDelta);
        hiddenBiasList[0] = hiddenBiasList[0].Add(gradient_Hidden);
        #endregion
    }


    //--------------------


    public void ResizeLayer(int nodesAmount, int index)
    {
        //Get a Bias
        if (index < hiddenBiasList.Count)
        {
            hiddenBiasList[index].AddNodes(nodesAmount,0);
        }

        //Adjust the weights
        if (index == 0)
        {
            hiddenWeights_Input.AddNodes(nodesAmount, 0);
            if (index < hiddenLayerList.Count - 1)
            {
                hiddenWeightList[index].AddNodes(0, nodesAmount);
            }
        }

        if (index == hiddenLayerList.Count - 1)
        {
            hiddenWeights_Output.AddNodes(0, nodesAmount);
            if (index > 0)
            {
                hiddenWeightList[index - 1].AddNodes(nodesAmount, 0);
            }
        }

        if (index > 0 && index < hiddenLayerList.Count - 1)
        {
            hiddenWeightList[index - 1].AddNodes(nodesAmount, 0);
            hiddenWeightList[index].AddNodes(0, nodesAmount);
        }

        //Adjust Hidden layers
        if (index < hiddenLayerList.Count)
        {
            hiddenLayerList[index].AddNodes(nodesAmount, 0);
        }
    }


    //--------------------


    public void AddHiddenLayer(int size)
    {
        //Add a hidden layer with a given size

        hiddenBiasList.Add(new Matrix(size, 1));
        hiddenBiasList[hiddenBiasList.Count - 1].Randomize();

        hiddenLayerList.Add(new Matrix(size, 1));
        hiddenLayerList[hiddenLayerList.Count - 1].Randomize();

        hiddenWeightList.Add(new Matrix(size, hiddenLayerList[hiddenLayerList.Count - 2].rows));
        hiddenWeightList[hiddenWeightList.Count -1].Randomize();

        hiddenWeights_Output = new Matrix(layer_Output.rows, size);
        hiddenWeights_Output.Randomize();
    }


    //--------------------


    public void Mutate()
    {
        //Make it possible to mutate the cars for the next generations, to give a chance of imporoved performance

        int mutationPropability = UnityEngine.Random.Range(10, 100);

        for (int n = 0; n < hiddenLayerList.Count; n++) 
        {
            for (int i = 0; i < hiddenLayerList[n].rows; i++)
            {
                int check = UnityEngine.Random.Range(0, 101);
                if (check < mutationPropability)
                {
                    hiddenLayerList[n].Mutate(i, 0, Random.Range(1, 25));
                }
            }
        }

        for (int n = 0; n < hiddenWeightList.Count; n++)
        {
            for (int i = 0; i < hiddenWeightList[n].rows; i++)
            {

                for (int j = 0; j < hiddenWeightList[n].cols; j++)
                {
                    int check = UnityEngine.Random.Range(0, 101);
                    if (check < mutationPropability)
                    {
                        hiddenWeightList[n].Mutate(i, j, Random.Range(1, 25));
                    }
                }
            }
        }

        for (int i = 0; i < hiddenWeights_Input.rows; i++)
        {

            for (int j = 0; j < hiddenWeights_Input.cols; j++)
            {
                int check = UnityEngine.Random.Range(0, 101);
                if (check < mutationPropability)
                {
                    hiddenWeights_Input.Mutate(i, j, Random.Range(1, 25));
                }
            }
        }

        for (int i = 0; i < hiddenWeights_Output.rows; i++)
        {

            for (int j = 0; j < hiddenWeights_Output.cols; j++)
            {
                int check = UnityEngine.Random.Range(0, 101);
                if (check < mutationPropability)
                {
                    hiddenWeights_Output.Mutate(i, j, Random.Range(1, 25));
                }
            }
        }

        for (int n = 0; n < hiddenBiasList.Count; n++)
        {
            for (int i = 0; i < hiddenBiasList[n].rows; i++)
            {

                for (int j = 0; j < hiddenBiasList[n].cols; j++)
                {
                    int check = UnityEngine.Random.Range(0, 101);
                    if (check < mutationPropability)
                    {
                        hiddenBiasList[n].Mutate(i, j, Random.Range(1, 25));
                    }
                }
            }
        }

        for (int i = 0; i < bias_Output.rows; i++)
        {

            for (int j = 0; j < bias_Output.cols; j++)
            {
                int check = UnityEngine.Random.Range(0, 101);
                if (check < mutationPropability)
                {
                    bias_Output.Mutate(i, j, Random.Range(1,25));
                }
            }
        }
    }


    //--------------------


    public NeuralNetwork Copy()
    {
        //Copy the abilities of a NeuronNetwork from a parent into a child

        NeuralNetwork networkToCopy = new NeuralNetwork();

        for (int i = 0; i < hiddenBiasList.Count; i++)
        {
            networkToCopy.hiddenBiasList.Add(new Matrix(hiddenBiasList[i].mat));
        }

        networkToCopy.bias_Output = new Matrix(bias_Output.mat);
        networkToCopy.hiddenWeights_Input = new Matrix(hiddenWeights_Input.mat);

        for (int i = 0; i < hiddenWeightList.Count; i++)
        {
            networkToCopy.hiddenWeightList.Add(new Matrix(hiddenWeightList[i].mat));
        }

        networkToCopy.hiddenWeights_Output = new Matrix(hiddenWeights_Output.mat);
        networkToCopy.layer_Imput = new Matrix(hiddenWeights_Output.mat);

        for (int i = 0; i < hiddenLayerList.Count; i++)
        {
            networkToCopy.hiddenLayerList.Add(new Matrix(hiddenLayerList[i].mat));
        }

        networkToCopy.layer_Output = new Matrix(hiddenWeights_Output.mat);

        networkToCopy.hiddenNodesList = hiddenNodesList;
        networkToCopy.inputNodes_Amount = inputNodes_Amount;
        networkToCopy.outputNodes_Amount = outputNodes_Amount;

        return networkToCopy;
    }
}
