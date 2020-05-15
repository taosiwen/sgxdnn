#ifndef DNN_HPP
#define DNN_HPP

#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <string>
#include <omp.h>

#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>

//Layer type
#define LAYER_RELU      0   //relu activation layer
#define LAYER_SIGMOID   1   //sigmoid activation layer
#define LAYER_TANH      2   //tanh activation layer
#define LAYER_LINEAR    3   //linear activation layer
#define LAYER_SOFTMAX   4   //softmax layer


//Loss type
#define LOSS_MEAN_SQUARE    0   //mean square loss
#define LOSS_CROSS_ENTROPY  1   //cross entropy loss

//Optimize type
#define OPTIMIZE_SGD        0   //SGD optimize
#define OPTIMIZE_MOMENTUM   1   //mementum optimize

#define RELU_SLOPE 0.1
#define MOMENTUM_ALPHA 0.2


using namespace std;
using namespace Eigen;

struct LayerInfo{
    int input;
    int output;
    int type;


    MatrixXf weight;
    VectorXf bias;
};



class DNN{
    public:
        //Constructor & Destructor
        DNN();
        DNN(string model_path);
        DNN(string model_path, string param_path);
        void Init();

        //User operation
        int AddLayer(unsigned int input, unsigned int output, unsigned int type);
        int AddLayer(LayerInfo new_Layer);

        int InitModel();
        int ShowModel();
        int SaveModel(string path);
        int LoadModel(string path);
        int SaveParam(string path);
        int LoadParam(string path);

        int Train_safe(MatrixXf sample, MatrixXf label);
        int Train_safe(MatrixXf sample, MatrixXf label, int batch_size);
        int Test_safe(MatrixXf sample, MatrixXf label);
        int Test_safe(MatrixXf sample, MatrixXf label, int batch_size);

        //Get Set operation
        LayerInfo GetLayer(int layer_id);
        MatrixXf GetNet(int layer_id);
        MatrixXf GetInputTensor();
        MatrixXf GetLabelTensor();
        MatrixXf GetOutputTensor();

        MatrixXf get_first_weight();
        MatrixXf get_first_delta();
        int GetOptimizeType();
        float GetLearnRate();
        float GetDecayRate();
        float GetTotalLoss();

        int set_safe_net(MatrixXf net);
        int SetLearnRate(float rate);
        int SetDecayRate(float rate);
        int SetLossType(int type);
        int SetOptimizeType(int type);

        
        int SetInputTensor(MatrixXf input_ts);
        
        int SetLabelTensor(MatrixXf label_ts);

        //Inner operation
        int FrontProp(unsigned int layer_id, MatrixXf input_ts);
        int FrontPropTotal();
        
        int BackProp(unsigned int layer_id, MatrixXf delta_ts);
        int BackPropTotal();

        int Optimize(int type);
        //int Optimize_safe(int type);
        
        MatrixXf Loss(MatrixXf &y_out, MatrixXf &y_label);
        
        float Activation(float x, int type);
        MatrixXf TensorActivation(MatrixXf x, int type);
        float ActivationDiff(float x, int type);
        MatrixXf TensorActivationDiff(MatrixXf x, int type);

        MatrixXf InitWeight(int rows, int cols);
        float Gaussrand(float exp, float var);
        int *ListShuffle(int total);
        int Swap(int *a, int *b);

    private:
        int total_layer;
        float learn_rate;
        float decay_rate;
        int loss_type;
        int optimize_type;

        vector <LayerInfo> layer;
        vector <MatrixXf> net;
        vector <MatrixXf> delta;

        vector <MatrixXf> weight_delta;
        vector <VectorXf> bias_delta;
        vector <MatrixXf> weight_rec;
        vector <VectorXf> bias_rec;
        
        MatrixXf input_tensor;
        MatrixXf label_tensor;

        MatrixXf safe_net;

};

DNN::DNN(){
    Init();
}

DNN::DNN(string model_path){
    Init();
    LoadModel(model_path);
    InitModel();
}

DNN::DNN(string model_path, string param_path){
    Init();
    LoadModel(model_path);
    LoadParam(param_path);
}

void DNN::Init(){
    total_layer = 0;
    learn_rate = 0.1;
    decay_rate = 0;
    loss_type = 0;
    optimize_type = 0;
    srand(time(0));
}

int DNN::AddLayer(unsigned int input, unsigned int output, unsigned int type){
    LayerInfo new_layer;
    new_layer.input = input;
    new_layer.output = output;
    new_layer.type = type;
    new_layer.weight = InitWeight(output, input);
    new_layer.bias = VectorXf::Zero(output);
    AddLayer(new_layer);
    
    return 0;
}

int DNN::AddLayer(LayerInfo new_layer){
    ++total_layer;
    layer.push_back(new_layer);

    MatrixXf temp;
    VectorXf temp2;
    net.push_back(temp);
    delta.push_back(temp);
    weight_delta.push_back(temp);
    bias_delta.push_back(temp2);

    MatrixXf w_dr = MatrixXf::Zero(new_layer.output, new_layer.input);
    VectorXf b_dr = VectorXf::Zero(new_layer.output);
    weight_rec.push_back(w_dr);
    bias_rec.push_back(b_dr);
    return 0;
}



int DNN::InitModel(){
    #pragma omp parallel for
    for(int i=0; i<layer.size(); ++i){
        layer[i].weight = InitWeight(layer[i].output, layer[i].input);
        layer[i].bias = VectorXf::Zero(layer[i].output);
    }
    return 0;
}

int DNN::ShowModel(){
    cout << "==============================" << endl;
    cout << "[Model Info]" << endl;
    cout << "Total Layer : " << total_layer << endl;
    cout << "Learn Rate  : " << learn_rate << endl;
    cout << "Decay Rate  : " << decay_rate << endl;

    switch(loss_type){
        case 0:
            cout << "Loss  Type  : Mean Square" << endl;
            break;
        case 1:
            cout << "Loss  Type  : Cross Entropy" << endl;
            break;
        default:
            cout << "Loss  Type  : Unknown" << endl;
    }

    cout << "==============================" << endl;
    cout << "[Layer Info]" << endl;
    
    for(int i=0; i<total_layer; ++i){
        cout << "------------------------------" << endl;
        cout << "<Layer " << i << ">" << endl;
        cout << "Input  : " << layer[i].input << endl;
        cout << "Output : " << layer[i].output << endl;

        switch(layer[i].type){
            case 0:
                cout << "Type   : ReLU" << endl;
                break;
            case 1:
                cout << "Type   : Sigmoid" << endl;
                break;
            case 2:
                cout << "Type   : Tanh" << endl;
                break;
            case 3:
                cout << "Type   : Linear" << endl;
                break;
            case 4:
                cout << "Type   : Softmax" << endl;
                break;
            default:
                cout << "Type   : Unknown" << endl;
        }
    }
    cout << "==============================" << endl;
    return 0;
}

int DNN::SaveModel(string path){
    FILE *fp;
    fp = fopen(path.c_str(), "wb");
    int buf;

    //Write model information
    buf = 16;
    fwrite(&buf, sizeof(int), 1, fp);   //chunk size
    fwrite(&total_layer, sizeof(int), 1, fp);   //total layer
    fwrite(&learn_rate, sizeof(float), 1, fp);  //learn rate
    fwrite(&decay_rate, sizeof(float), 1, fp);  //decay rate
    fwrite(&loss_type, sizeof(int), 1, fp); //loss type
    
    //write layer information
    for(int i=0; i<total_layer; ++i){
        buf = 12;
        fwrite(&buf, sizeof(int), 1, fp);   //chunk size
        fwrite(&(layer[i].input), sizeof(int), 1, fp);  //input
        fwrite(&(layer[i].output), sizeof(int), 1, fp); //output
        fwrite(&(layer[i].type), sizeof(int), 1, fp);   //type
    }

    fclose(fp);
    return 0;
}

int DNN::LoadModel(string path){
    FILE *fp;
    fp = fopen(path.c_str(), "rb");
    int buf;
    int total_layer_buf;

    //Read model information
    fread(&buf, sizeof(int), 1, fp);            //chunk size
    fread(&total_layer_buf, sizeof(int), 1, fp);    //total layer
    fread(&learn_rate, sizeof(float), 1, fp);   //learn rate
    fread(&decay_rate, sizeof(float), 1, fp);   //decay rate
    fread(&loss_type, sizeof(int), 1, fp);      //loss type
    //cout << total_layer_buf << " " << learn_rate << " " << decay_rate << " " << loss_type << endl;
    
    //Read layer information
    int input_buf, output_buf, type_buf;
    for(int i=0; i<total_layer_buf; ++i){
        fread(&buf, sizeof(int), 1, fp);        //chunk size
        fread(&input_buf, sizeof(int), 1, fp);  //input
        fread(&output_buf, sizeof(int), 1, fp); //output
        fread(&type_buf, sizeof(int), 1, fp);   //type
        AddLayer(input_buf, output_buf, type_buf);
    }
    fclose(fp);
    return 0;
}

int DNN::SaveParam(string path){
    FILE *fp;
    fp = fopen(path.c_str(), "wb");
    fwrite(&total_layer, sizeof(int), 1, fp);   //total layer
    for(int i=0; i<total_layer; ++i){
        fwrite(&(layer[i].output), sizeof(int), 1, fp); //rows
        fwrite(&(layer[i].input), sizeof(int), 1, fp);  //cols

        int w_rows = layer[i].weight.rows();
        int w_cols = layer[i].weight.cols();
        
        fwrite(layer[i].weight.data(), sizeof(float), w_rows*w_cols, fp);   //weight
        fwrite(layer[i].bias.data(), sizeof(float), w_rows, fp);    //bias
    }

    fclose(fp);
    return 0;
}

int DNN::LoadParam(string path){
    FILE *fp;
    fp = fopen(path.c_str(), "rb");
    
    int buf;
    fread(&buf, sizeof(int), 1, fp);    //total layer
    InitModel();

    int w_rows, w_cols;
    for(int i=0; i<total_layer; ++i){
        fread(&w_rows, sizeof(int), 1, fp); //input
        fread(&w_cols, sizeof(int), 1, fp); //output

        for(int y=0; y<w_cols; ++y)
            for(int x=0; x<w_rows; ++x)
                fread(&(layer[i].weight(x,y)), sizeof(float), 1, fp);   //weight

        for(int x=0; x<w_rows; ++x)
            fread(&(layer[i].bias(x)), sizeof(float), 1, fp);   //bias
        
        //layer[i].weight = Map<MatrixXf>((float *)w_buf, w_rows, w_cols);
    }


    fclose(fp);
    return 0;
}

int DNN::Train_safe(MatrixXf sample, MatrixXf label){

    input_tensor = sample;
    label_tensor = label;
    return 0;
}

int DNN::Train_safe(MatrixXf sample, MatrixXf label, int batch_size){
    int total_size = sample.cols();
    int *list = ListShuffle(total_size);
    MatrixXf sample_batch(sample.rows(), batch_size);
    MatrixXf label_batch(label.rows(), batch_size);
    for(int i=0; i<batch_size; ++i){
        sample_batch.col(i) = sample.col(list[i]);
        label_batch.col(i) = label.col(list[i]);
    }
    delete [] list;
    input_tensor = sample_batch;
    label_tensor = label_batch;
    return 0;
}
int DNN::Test_safe(MatrixXf sample, MatrixXf label){
    input_tensor = sample;
    label_tensor = label;
    return 0;
}

int DNN::Test_safe(MatrixXf sample, MatrixXf label, int batch_size){
    int total_size = sample.cols();
    int *list = ListShuffle(total_size);
    MatrixXf sample_batch(sample.rows(), batch_size);
    MatrixXf label_batch(label.rows(), batch_size);
    for(int i=0; i<batch_size; ++i){
        sample_batch.col(i) = sample.col(list[i]);
        label_batch.col(i) = label.col(list[i]);
    }
    delete [] list;
    input_tensor = sample_batch;
    label_tensor = label_batch;
    return 0;
}

LayerInfo DNN::GetLayer(int layer_id){
    return layer[layer_id];
}

MatrixXf DNN::GetNet(int layer_id){
    return net[layer_id];
}

MatrixXf DNN::GetInputTensor(){
    return input_tensor;
}

MatrixXf DNN::GetLabelTensor(){
    return label_tensor;
}

MatrixXf DNN::GetOutputTensor(){
    return net[total_layer-1];
}

MatrixXf DNN::get_first_weight(){
    return layer[0].weight;
}

MatrixXf DNN::get_first_delta(){
    return delta[0];
}

int DNN::GetOptimizeType(){
    return optimize_type;
}

float DNN::GetLearnRate(){
    return learn_rate;
}
float DNN::GetDecayRate(){
    return decay_rate;
}

float DNN::GetTotalLoss(){
    float err = 0;
    #pragma omp parallel for
    for(int i=0; i<label_tensor.rows(); ++i)
        for(int j=0; j<label_tensor.cols(); ++j)
            err += pow((label_tensor(i,j) - net[total_layer-1](i,j)), 2);
    err /= (label_tensor.rows()*label_tensor.cols());
    return err;
}

int DNN::SetLearnRate(float rate){
    learn_rate = rate;
    return 0;
}

int DNN::set_safe_net(MatrixXf net){
    safe_net = net;
    return 0;
}

int DNN::SetDecayRate(float rate){
    decay_rate = rate;
    return 0;
}

int DNN::SetLossType(int type){
    loss_type = type;
    return 0;
}

int DNN::SetOptimizeType(int type){
    optimize_type = type;
    return 0;
}

int DNN::SetInputTensor(MatrixXf input_ts){
    input_tensor = input_ts;
    return 0;
}

int DNN::SetLabelTensor(MatrixXf label_ts){
    label_tensor = label_ts;
    return 0;
}

int DNN::FrontProp(unsigned int layer_id, MatrixXf input_ts){

    net[layer_id] = layer[layer_id].weight*input_ts;

    #pragma omp parallel for
    for(int i=0; i<input_ts.cols(); ++i)
        net[layer_id].col(i) += layer[layer_id].bias;

    net[layer_id] = TensorActivation(net[layer_id], layer[layer_id].type);

    return 0;
}


int DNN::FrontPropTotal(){
    FrontProp(0, safe_net);
    for(int i=1; i<layer.size(); ++i)
        FrontProp(i, net[i-1]);
    return 0;
}


int DNN::BackProp(unsigned int layer_id, MatrixXf delta_ts){
    MatrixXf delta_temp;

    if(layer_id == total_layer-1)
        delta_temp = Loss(net[total_layer-1], delta_ts);
    else
        delta_temp = layer[layer_id+1].weight.transpose()*delta_ts;

    MatrixXf TsActDiff = TensorActivationDiff(net[layer_id], layer[layer_id].type);
    delta[layer_id] = delta_temp.cwiseProduct(TsActDiff);

    return 0;
}
    

int DNN::BackPropTotal(){
    BackProp(total_layer-1, label_tensor);
    for(int i=total_layer-2; i>=0; --i)
        BackProp(i, delta[i+1]);

    for(int i=total_layer-1; i>=0; --i){
        weight_delta[i] = MatrixXf::Zero(layer[i].weight.rows(), layer[i].weight.cols());
        bias_delta[i] = VectorXf::Zero(layer[i].bias.rows());

        for(int j=0; j<label_tensor.cols(); ++j){
            if(i != 0)
                weight_delta[i] += delta[i].col(j) * net[i-1].col(j).transpose();
            else
                weight_delta[i] += delta[i].col(j) * safe_net.col(j).transpose();
            bias_delta[i] += delta[i].col(j);
        }

        weight_delta[i] /= label_tensor.cols();
        bias_delta[i] /= label_tensor.cols();   
    }
}



int DNN::Optimize(int type){
    for(int i=0; i<total_layer; ++i){
        switch(type){
            case 0:
                layer[i].weight -= learn_rate*(weight_delta[i] + decay_rate*layer[i].weight);
                layer[i].bias -= learn_rate*bias_delta[i];
                break;
            case 1:
                float alpha = MOMENTUM_ALPHA;
                MatrixXf weight_learn = weight_delta[i] + decay_rate*layer[i].weight;
                VectorXf bias_learn = bias_delta[i];

                weight_rec[i] = weight_learn + alpha*weight_rec[i];
                bias_rec[i] = bias_learn + alpha*bias_rec[i];
                //weight_rec[i] = (1-alpha)*weight_learn + alpha*weight_rec[i];
                //bias_rec[i] = (1-alpha)*bias_learn + alpha*bias_rec[i];


                layer[i].weight -= learn_rate*weight_rec[i];
                layer[i].bias -= learn_rate*bias_rec[i];
                break;
        }
    }
    return 0;
}

/*
int DNN::Optimize_safe(int type){
    switch(type){
        case 0:
            safe_layer.weight -= learn_rate*(safe_weight_delta+decay_rate*safe_layer.weight);
            break;

        case 1:
            float alpha = MOMENTUM_ALPHA;
            MatrixXf weight_learn = safe_weight_delta + decay_rate*safe_layer.weight;

            safe_weight_rec = weight_learn + alpha*safe_weight_rec;
            
            safe_layer.weight -= learn_rate*safe_weight_rec;

            break;
    }

    return 0;
}
*/


MatrixXf DNN::Loss(MatrixXf &y_out, MatrixXf &y_label){
    MatrixXf delta_temp;
    switch(loss_type){
        case 0:     //mean square loss
            delta_temp = -(y_label - y_out);
            break;

        case 1: //cross entropy loss
            delta_temp = -(y_label - y_out);
            #pragma omp parallel for
            for(int i=0; i<y_out.rows(); ++i)
                for(int j=0; j<y_out.cols(); ++j)
                    delta_temp(i,j) /= (y_out(i,j)*(1-y_out(i,j)));
            break;

        default:
            delta_temp = -(y_label - y_out);
    }
    
    return delta_temp;
}

float DNN::Activation(float x, int type){
    float y;
    switch(type){
        case 0:
            y = (x <= 0)? RELU_SLOPE*x:x;
            break;
        case 1:
            y = 1/(1+exp(-x));
            break;
        case 2:
            y = tanh(x);
            break;
        case 3:
            y = x;
            break;
        default:
            y = 1/(1+exp(-x));
    }
    return y;
}

MatrixXf DNN::TensorActivation(MatrixXf x, int type){
    MatrixXf y(x.rows(), x.cols());

    if(type == 4){
        for(int j=0; j<x.cols(); ++j){
            for(int i=0; i<x.rows(); ++i)
                y(i,j) = exp(x(i,j)-x.col(j).maxCoeff());
            y.col(j) /= y.col(j).sum();
        }
    }

    #pragma omp parallel for
    for(int i=0; i<x.rows(); ++i)
        for(int j=0; j<x.cols(); ++j)
            y(i,j) = Activation(x(i,j), type);

    return y;
}

float DNN::ActivationDiff(float x, int type){
    float y;
    switch(type){
        case 0:
            y = (x <= 0)? RELU_SLOPE:1;
            break;
        case 1:
            y = x * (1 - x);
            break;
        case 2:
            y = 1 - pow(x,2);
            break;
        case 3:
            y = 1;
            break;
        default:
            y = x * (1 - x);
    }
    return y;
}

MatrixXf DNN::TensorActivationDiff(MatrixXf x, int type){
    MatrixXf y(x.rows(), x.cols());

    #pragma omp parallel for
    for(int i=0; i<x.rows(); ++i)
        for(int j=0; j<x.cols(); ++j)
            y(i,j) = ActivationDiff(x(i,j), type);

    return y;
}


MatrixXf DNN::InitWeight(int rows, int cols){
    MatrixXf temp(rows, cols);
    #pragma omp parallel for
    for(int i=0; i<rows; ++i)
        for(int j=0; j<cols; ++j)
            temp(i,j) = Gaussrand(0, 1/sqrt(rows));

    return temp;
}



float DNN::Gaussrand(float exp, float var){
    static float V1, V2, S;
    static int phase = 0;
    float X;

    if(phase == 0){
        do {
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;
 
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);
    }
    else
        X = V2 * sqrt(-2 * log(S) / S);
    
    phase = 1 - phase;
    X = X * var + exp;
    return X;
}

int *DNN::ListShuffle(int total){
    int *list = new int[total];
    #pragma omp parallel for
    for(int i=0; i<total; ++i)
        list[i] = i;
    for(int i=total-1; i>0; --i){
        int temp = rand() % i;
        Swap(&(list[i]), &(list[temp]));
    }
    return list;
}

int DNN::Swap(int *a, int *b){
    int c = *b;
    *b = *a;
    *a = c;
    return 0;
}

#endif // DNN_HPP
